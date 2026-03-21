"""Repair operators for track layout chromosomes.

Repair operators run post-mutation, pre-evaluation to fix constraint violations.

IMPORTANT: All operators use DECODE-MODIFY-RE-ENCODE pattern because:
- Chromosomes use Random-Key (RK) encoding with values [0.0, 1.0]
- Operators cannot directly interpret RK values as piece indices
- Must decode to get actual piece sequence, repair, then re-encode

Operators:
- InventoryRepair: Remove pieces exceeding inventory limits
- ClosureRepair: Attempt to fix angular closure (basic implementation)
- TrackRepairPipeline: Combines all repairs in sequence

Baldwinian by default: evaluate repaired phenotype, keep original genotype.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pymoo.core.repair import Repair

from .encoding import (
    INACTIVE,
    L_MAX,
    MAIN_LOOP_END,
    MAIN_LOOP_START,
    N_VAR,
    RK_INACTIVE_THRESHOLD,
    STRAIGHT_16,
    SWITCH_LEFT_IN,
    SWITCH_LEFT_OUT,
    SWITCH_PAIRS,
    SWITCH_RIGHT_IN,
    SWITCH_RIGHT_OUT,
    create_chromosome_from_pattern,
    get_main_loop,
    get_piece_keys,
    piece_index_to_rk,
    rk_to_piece_index,
)


# =============================================================================
# RK Encoding Helpers
# =============================================================================

def _decode_rk_to_pieces(
    x: NDArray,
    inventory_by_index: Dict[int, int],
) -> Tuple[List[int], List[int]]:
    """Decode RK chromosome to piece sequence.

    Simulates the decoder's RK-to-piece mapping to get actual piece indices.

    Args:
        x: Chromosome array with RK values [0, 1].
        inventory_by_index: Available inventory {piece_index: count}.

    Returns:
        Tuple of (piece_indices, active_positions) - pieces decoded and their positions.
    """
    piece_keys = x[MAIN_LOOP_START:MAIN_LOOP_END]
    pieces = []
    positions = []
    inventory_used: Dict[int, int] = {}

    for pos, rk_value in enumerate(piece_keys):
        if rk_value < RK_INACTIVE_THRESHOLD:
            continue

        # Build available pieces from remaining inventory (sorted like decoder)
        available = []
        for idx in sorted(inventory_by_index.keys()):
            remaining = inventory_by_index[idx] - inventory_used.get(idx, 0)
            if remaining > 0:
                available.append(idx)

        if not available:
            break

        piece_idx = rk_to_piece_index(float(rk_value), available)
        if piece_idx >= 0:
            pieces.append(piece_idx)
            positions.append(pos)
            inventory_used[piece_idx] = inventory_used.get(piece_idx, 0) + 1

    return pieces, positions


def _encode_pieces_to_rk(
    pieces: List[int],
    available_pieces: List[int],
    original_x: NDArray,
) -> NDArray:
    """Re-encode piece sequence as RK chromosome.

    Preserves non-main-loop genes from original chromosome.

    Args:
        pieces: List of piece indices to encode.
        available_pieces: List of available piece indices for RK mapping.
        original_x: Original chromosome (for preserving other genes).

    Returns:
        New chromosome with encoded pieces.
    """
    x = original_x.copy()

    # Clear main loop
    x[MAIN_LOOP_START:MAIN_LOOP_END] = 0.0  # Inactive

    # Encode pieces at sequential positions
    for i, piece_idx in enumerate(pieces):
        if i >= L_MAX:
            break
        rk_value = piece_index_to_rk(piece_idx, available_pieces)
        x[MAIN_LOOP_START + i] = max(RK_INACTIVE_THRESHOLD + 0.001, rk_value)

    return x


# =============================================================================
# Inventory Repair
# =============================================================================

class InventoryRepair(Repair):
    """Remove pieces exceeding inventory limits using RK-aware decode-modify-re-encode.

    Process:
    1. Decode RK chromosome to piece sequence
    2. Count piece usage and remove excess (prefer removing later pieces)
    3. Re-encode repaired sequence as RK chromosome
    """

    def __init__(
        self,
        inventory_by_index: Dict[int, int],
        index_to_id: Optional[Dict[int, str]] = None,
    ):
        """Initialize inventory repair.

        Args:
            inventory_by_index: Available inventory {piece_index: count}.
            index_to_id: Mapping from piece index to ID (for validation).
        """
        super().__init__()
        self.inventory_by_index = inventory_by_index
        self.index_to_id = index_to_id or {}
        # Build sorted available pieces list for RK encoding
        self.available_pieces = sorted(inventory_by_index.keys())

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Repair population to satisfy inventory limits.

        Args:
            problem: Optimization problem.
            X: Population array (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Repaired population array.
        """
        X_repaired = X.copy()

        for i in range(len(X)):
            X_repaired[i] = self._repair_chromosome(X[i])

        return X_repaired

    def _repair_chromosome(self, x: NDArray) -> NDArray:
        """Repair single chromosome using decode-modify-re-encode.

        Args:
            x: Chromosome array with RK values.

        Returns:
            Repaired chromosome.
        """
        # Step 1: Decode RK to piece sequence
        pieces, _ = _decode_rk_to_pieces(x, self.inventory_by_index)

        if not pieces:
            return x  # Nothing to repair

        # Step 2: Count usage and remove excess
        usage: Dict[int, int] = {}
        repaired_pieces = []

        for piece_idx in pieces:
            used = usage.get(piece_idx, 0)
            available = self.inventory_by_index.get(piece_idx, 0)

            if used < available:
                repaired_pieces.append(piece_idx)
                usage[piece_idx] = used + 1
            # else: skip this piece (over inventory limit)

        # Step 3: Re-encode as RK chromosome
        return _encode_pieces_to_rk(repaired_pieces, self.available_pieces, x)


# =============================================================================
# Closure Repair (Basic) - RK-Aware
# =============================================================================

class ClosureRepair(Repair):
    """Attempt to fix angular closure violations using RK-aware decode-modify-re-encode.

    Process:
    1. Decode RK chromosome to piece sequence
    2. Compute angular deficit (360 - total_angle)
    3. If deficit > 0 (need more turning), add curves
    4. If deficit < 0 (too much turning), remove curves
    5. Re-encode repaired sequence as RK chromosome
    """

    def __init__(
        self,
        catalog,
        inventory_by_index: Optional[Dict[int, int]] = None,
        curve_piece_index: int = 2,  # R40_LEFT default
    ):
        """Initialize closure repair.

        Args:
            catalog: Track catalog.
            inventory_by_index: Available inventory {piece_index: count}.
            curve_piece_index: Piece index for typical curve.
        """
        super().__init__()
        self.catalog = catalog
        self.curve_piece_index = curve_piece_index
        self.inventory_by_index = inventory_by_index or {}

        # Build sorted available pieces list for RK encoding
        self.available_pieces = sorted(self.inventory_by_index.keys()) if self.inventory_by_index else list(range(10))

        # Get angle per curve piece
        fk = catalog._fk_table[curve_piece_index]
        self.curve_angle = abs(fk[2])  # degrees

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Repair population for closure.

        Args:
            problem: Optimization problem.
            X: Population array (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Repaired population array.
        """
        X_repaired = X.copy()

        for i in range(len(X)):
            X_repaired[i] = self._repair_chromosome(X[i])

        return X_repaired

    def _repair_chromosome(self, x: NDArray) -> NDArray:
        """Repair single chromosome for closure using decode-modify-re-encode.

        Args:
            x: Chromosome array with RK values.

        Returns:
            Repaired chromosome.
        """
        # Step 1: Decode RK to piece sequence
        pieces, _ = _decode_rk_to_pieces(x, self.inventory_by_index)

        if not pieces:
            return x  # Nothing to repair

        # Step 2: Compute total angle
        total_angle = sum(
            abs(self.catalog._fk_table[idx, 2])
            for idx in pieces
            if idx < len(self.catalog._fk_table)
        )

        # Compute deficit (positive = need more turning)
        deficit = 360.0 - total_angle

        # Step 3: Modify pieces to fix closure
        repaired_pieces = list(pieces)

        if deficit > self.curve_angle:
            # Need more turning - add curves (up to 4)
            n_curves_to_add = min(int(np.ceil(deficit / self.curve_angle)), 4)
            for _ in range(n_curves_to_add):
                repaired_pieces.append(self.curve_piece_index)

        elif deficit < -self.curve_angle:
            # Too much turning - remove curves from end (keep at least 4 pieces)
            n_to_remove = min(int(np.ceil(-deficit / self.curve_angle)), len(repaired_pieces) - 4)

            removed = 0
            while removed < n_to_remove and len(repaired_pieces) > 4:
                # Find last curve and remove it
                for i in range(len(repaired_pieces) - 1, -1, -1):
                    piece_angle = abs(self.catalog._fk_table[repaired_pieces[i], 2])
                    if piece_angle > 0:  # Is a curve
                        repaired_pieces.pop(i)
                        removed += 1
                        break
                else:
                    break  # No more curves to remove

        # Step 4: Re-encode as RK chromosome
        return _encode_pieces_to_rk(repaired_pieces, self.available_pieces, x)


# =============================================================================
# Switch Repair (Balance Pairs) - RK-Aware
# =============================================================================

class SwitchRepair(Repair):
    """Balance switch pairs using RK-aware decode-modify-re-encode.

    An orphaned switch is one that cannot form a valid switch pair:
    - LEFT_IN without matching LEFT_OUT (or vice versa)
    - RIGHT_IN without matching RIGHT_OUT (or vice versa)

    Process:
    1. Decode RK chromosome to piece sequence
    2. Count IN and OUT switches for each handedness
    3. Balance pairs by adding missing or removing excess
    4. Re-encode repaired sequence as RK chromosome
    """

    def __init__(
        self,
        inventory_by_index: Optional[Dict[int, int]] = None,
        prefer_add: bool = True,
    ):
        """Initialize switch repair.

        Args:
            inventory_by_index: Available inventory {piece_index: count}.
                If provided, enables adding missing switches within inventory.
            prefer_add: If True, prefer adding missing switches over removing excess.
        """
        super().__init__()
        self.inventory_by_index = inventory_by_index or {}
        self.prefer_add = prefer_add
        # Build sorted available pieces list for RK encoding
        self.available_pieces = sorted(self.inventory_by_index.keys()) if self.inventory_by_index else list(range(10))

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Repair population to balance switch pairs.

        Args:
            problem: Optimization problem.
            X: Population array (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Repaired population array.
        """
        X_repaired = X.copy()

        for i in range(len(X)):
            X_repaired[i] = self._repair_chromosome(X[i])

        return X_repaired

    def _repair_chromosome(self, x: NDArray) -> NDArray:
        """Repair single chromosome by balancing switch pairs.

        Uses decode-modify-re-encode pattern:
        1. Decode to piece sequence
        2. Balance switches
        3. Re-encode as RK chromosome

        Args:
            x: Chromosome array with RK values.

        Returns:
            Repaired chromosome.
        """
        # Step 1: Decode RK to piece sequence
        pieces, _ = _decode_rk_to_pieces(x, self.inventory_by_index)

        if not pieces:
            return x  # Nothing to repair

        # Step 2: Count switches and straights
        left_in_indices = [i for i, p in enumerate(pieces) if p == SWITCH_LEFT_IN]
        left_out_indices = [i for i, p in enumerate(pieces) if p == SWITCH_LEFT_OUT]
        right_in_indices = [i for i, p in enumerate(pieces) if p == SWITCH_RIGHT_IN]
        right_out_indices = [i for i, p in enumerate(pieces) if p == SWITCH_RIGHT_OUT]
        straight_indices = [i for i, p in enumerate(pieces) if p == STRAIGHT_16]

        # Track usage for inventory check
        usage = {}
        for p in pieces:
            usage[p] = usage.get(p, 0) + 1

        repaired_pieces = list(pieces)

        # Balance LEFT switches
        repaired_pieces, usage = self._balance_switch_pair_pieces(
            repaired_pieces, left_in_indices, left_out_indices,
            straight_indices, SWITCH_LEFT_IN, SWITCH_LEFT_OUT, usage
        )

        # Recount straights
        straight_indices = [i for i, p in enumerate(repaired_pieces) if p == STRAIGHT_16]

        # Balance RIGHT switches
        repaired_pieces, usage = self._balance_switch_pair_pieces(
            repaired_pieces, right_in_indices, right_out_indices,
            straight_indices, SWITCH_RIGHT_IN, SWITCH_RIGHT_OUT, usage
        )

        # Step 3: Re-encode as RK chromosome
        return _encode_pieces_to_rk(repaired_pieces, self.available_pieces, x)

    def _can_add_piece(self, piece_idx: int, usage: Dict[int, int]) -> bool:
        """Check if we can add a piece within inventory limits."""
        if not self.inventory_by_index:
            return False
        available = self.inventory_by_index.get(piece_idx, 0)
        used = usage.get(piece_idx, 0)
        return used < available

    def _balance_switch_pair_pieces(
        self,
        pieces: List[int],
        in_indices: List[int],
        out_indices: List[int],
        straight_indices: List[int],
        in_piece_idx: int,
        out_piece_idx: int,
        usage: Dict[int, int],
    ) -> Tuple[List[int], Dict[int, int]]:
        """Balance IN and OUT switch counts in piece list.

        Args:
            pieces: Piece sequence to modify.
            in_indices: Indices of IN switches in pieces.
            out_indices: Indices of OUT switches in pieces.
            straight_indices: Indices of straights in pieces.
            in_piece_idx: Piece index for IN switch.
            out_piece_idx: Piece index for OUT switch.
            usage: Current piece usage counts.

        Returns:
            Tuple of (modified pieces, updated usage).
        """
        n_in = len(in_indices)
        n_out = len(out_indices)

        if n_in == n_out:
            return pieces, usage  # Already balanced

        if n_in > n_out:
            # Need more OUT switches
            deficit = n_in - n_out

            if self.prefer_add:
                # Try to convert straights to OUT switches
                added = 0
                for idx in straight_indices:
                    if added >= deficit:
                        break
                    if not self._can_add_piece(out_piece_idx, usage):
                        break
                    # Check if there's an IN switch before this position
                    if any(in_idx < idx for in_idx in in_indices):
                        pieces[idx] = out_piece_idx
                        usage[out_piece_idx] = usage.get(out_piece_idx, 0) + 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 1) - 1
                        added += 1

                # Remove remaining excess IN switches
                remaining = deficit - added
                for i in range(remaining):
                    if in_indices:
                        idx = in_indices.pop()
                        pieces[idx] = STRAIGHT_16
                        usage[in_piece_idx] = usage.get(in_piece_idx, 1) - 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 0) + 1
            else:
                # Remove excess IN switches
                for i in range(deficit):
                    if in_indices:
                        idx = in_indices.pop()
                        pieces[idx] = STRAIGHT_16
                        usage[in_piece_idx] = usage.get(in_piece_idx, 1) - 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 0) + 1

        elif n_out > n_in:
            # Need more IN switches
            deficit = n_out - n_in

            if self.prefer_add:
                # Try to convert straights to IN switches
                added = 0
                for idx in straight_indices:
                    if added >= deficit:
                        break
                    if not self._can_add_piece(in_piece_idx, usage):
                        break
                    # Check if there's an OUT switch after this position
                    if any(out_idx > idx for out_idx in out_indices):
                        pieces[idx] = in_piece_idx
                        usage[in_piece_idx] = usage.get(in_piece_idx, 0) + 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 1) - 1
                        added += 1

                # Remove remaining excess OUT switches
                remaining = deficit - added
                for i in range(remaining):
                    if out_indices:
                        idx = out_indices.pop()
                        pieces[idx] = STRAIGHT_16
                        usage[out_piece_idx] = usage.get(out_piece_idx, 1) - 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 0) + 1
            else:
                # Remove excess OUT switches
                for i in range(deficit):
                    if out_indices:
                        idx = out_indices.pop()
                        pieces[idx] = STRAIGHT_16
                        usage[out_piece_idx] = usage.get(out_piece_idx, 1) - 1
                        usage[STRAIGHT_16] = usage.get(STRAIGHT_16, 0) + 1

        return pieces, usage


# =============================================================================
# Loose Port Repair
# =============================================================================

class LoosePortRepair(Repair):
    """Repair unpaired switches by removing them from the main loop.

    Switches have 3 ports. Port 0 and port 1 connect sequentially in the
    main loop. Port 2 (diverging) needs a branch to connect. Unpaired
    switches — those without a matching partner at the right geometric
    distance — create loose ports that violate constraint G[4].

    This repair scans for switches that can't form valid pairs and replaces
    them with STRAIGHT_16, ensuring loose_port_count = 0.

    Valid pairs (IN followed by OUT of same handedness at compatible distance)
    are preserved. The decoder's _extract_switch_pairs handles branch creation.
    """

    def __init__(self, inventory_by_index: Optional[Dict[int, int]] = None):
        super().__init__()
        self.inventory_by_index = inventory_by_index or {}
        self.available_pieces = sorted(self.inventory_by_index.keys()) if self.inventory_by_index else list(range(10))

    def _do(self, problem, X, **kwargs) -> NDArray:
        X_repaired = X.copy()
        for i in range(len(X)):
            X_repaired[i] = self._repair_chromosome(X[i])
        return X_repaired

    def _repair_chromosome(self, x: NDArray) -> NDArray:
        """Remove unpaired switches from main loop.

        For each handedness (LEFT, RIGHT):
        1. Find all IN and OUT positions
        2. Greedily pair IN→OUT (IN must come before OUT)
        3. Replace unpaired switches with STRAIGHT_16
        """
        pieces, positions = _decode_rk_to_pieces(x, self.inventory_by_index)
        if not pieces:
            return x

        repaired = list(pieces)
        paired_positions = set()

        # Pair switches for each handedness
        for in_idx, out_idx in SWITCH_PAIRS:
            in_positions = [i for i, p in enumerate(repaired) if p == in_idx]
            out_positions = [i for i, p in enumerate(repaired) if p == out_idx]

            # Greedy pairing: each IN pairs with nearest downstream OUT
            used_outs = set()
            for in_pos in sorted(in_positions):
                best_out = None
                for out_pos in sorted(out_positions):
                    if out_pos > in_pos and out_pos not in used_outs:
                        best_out = out_pos
                        break
                if best_out is not None:
                    paired_positions.add(in_pos)
                    paired_positions.add(best_out)
                    used_outs.add(best_out)

            # Remove unpaired IN switches
            for pos in in_positions:
                if pos not in paired_positions:
                    repaired[pos] = STRAIGHT_16

            # Remove unpaired OUT switches
            for pos in out_positions:
                if pos not in paired_positions:
                    repaired[pos] = STRAIGHT_16

        # Re-encode if any changes made
        if repaired != list(pieces):
            return _encode_pieces_to_rk(repaired, self.available_pieces, x)
        return x


# =============================================================================
# Combined Repair Pipeline
# =============================================================================

class TrackRepairPipeline(Repair):
    """Combined repair pipeline running multiple repairs in sequence.

    Order:
    1. Switch repair (balance switch pairs by adding/removing)
    2. Inventory repair (remove excess pieces)
    3. Closure repair (fix angular closure)
    """

    def __init__(
        self,
        catalog,
        inventory: Dict[str, int],
        enable_closure_repair: bool = True,
        enable_switch_repair: bool = True,
        prefer_add_switches: bool = True,
    ):
        """Initialize repair pipeline.

        Args:
            catalog: Track catalog.
            inventory: Available inventory {piece_id: count}.
            enable_closure_repair: Whether to apply closure repair.
            enable_switch_repair: Whether to apply switch repair.
            prefer_add_switches: If True, prefer adding missing switches over
                removing excess (helps preserve complex solutions).
        """
        super().__init__()

        # Convert inventory to index-based
        inventory_by_index = {}
        index_to_id = {}
        for piece_id, count in inventory.items():
            idx = catalog._id_to_index.get(piece_id)
            if idx is not None:
                inventory_by_index[idx] = count
                index_to_id[idx] = piece_id

        self.switch_repair = SwitchRepair(
            inventory_by_index=inventory_by_index,
            prefer_add=prefer_add_switches,
        ) if enable_switch_repair else None
        self.loose_port_repair = LoosePortRepair(inventory_by_index=inventory_by_index)
        self.inventory_repair = InventoryRepair(inventory_by_index, index_to_id)
        self.closure_repair = ClosureRepair(
            catalog,
            inventory_by_index=inventory_by_index,
        ) if enable_closure_repair else None

    def _do(self, problem, X, **kwargs) -> NDArray:
        """Apply repair pipeline to population.

        Args:
            problem: Optimization problem.
            X: Population array (n_pop, n_var).
            **kwargs: Additional arguments.

        Returns:
            Repaired population array.
        """
        # Step 1: Switch repair (balance IN/OUT counts)
        if self.switch_repair is not None:
            X = self.switch_repair._do(problem, X, **kwargs)

        # Step 2: Loose port repair DISABLED — switches evolve via soft
        # constraint penalty instead of being removed pre-evaluation.
        # X = self.loose_port_repair._do(problem, X, **kwargs)

        # Step 3: Inventory repair
        X = self.inventory_repair._do(problem, X, **kwargs)

        # Step 3: Closure repair (optional)
        if self.closure_repair is not None:
            X = self.closure_repair._do(problem, X, **kwargs)

        return X


# =============================================================================
# Utility Functions
# =============================================================================

def compute_piece_usage(x: NDArray) -> Dict[int, int]:
    """Count usage of each piece type in main loop.

    Args:
        x: Chromosome array.

    Returns:
        Dict mapping piece_index to count.
    """
    main_loop = get_main_loop(x)
    usage: Dict[int, int] = {}

    for gene in main_loop:
        gene_val = int(gene)
        if gene_val >= 0:
            usage[gene_val] = usage.get(gene_val, 0) + 1

    return usage


def validate_inventory(
    x: NDArray,
    inventory_by_index: Dict[int, int],
) -> bool:
    """Check if chromosome satisfies inventory limits.

    Args:
        x: Chromosome array.
        inventory_by_index: Available inventory {piece_index: count}.

    Returns:
        True if valid, False if inventory exceeded.
    """
    usage = compute_piece_usage(x)

    for piece_idx, count in usage.items():
        available = inventory_by_index.get(piece_idx, 0)
        if count > available:
            return False

    return True

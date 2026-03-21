# Przewodnik Techniczny: Algorytm Genetyczny Optymalizacji Torów LEGO (11.py)

---

## Część 1: Wstęp i Architektura

### Problem

Znaleźć zamkniętą pętlę toru LEGO używając maksymalnej liczby elementów z magazynu, mieszcząc się w granicach przestrzeni.

### Serce Algorytmu

Trzy mechanizmy współpracujące ze sobą, każdy adresuje inny aspekt problemu:

#### 1. MUTATION-ONLY (brak krzyżowania)

**Dlaczego?** Tory budowane są SEKWENCYJNIE. Każdy element zależy od poprzednich.
Krzyżowanie (zamiana fragmentów między torami) niszczyła te zależności dlatego zostało usunięte.
Pozycja każdego elementu jest obliczana przez forward propagation - zmiana w środku toru zmienia WSZYSTKO co następuje.

**Funkcje klasy MutationOperator** (linie 1198-1933):

- **`__init__`** (1208-1218)
  - Ustawia magazyn elementów, granice i etap programu
  - Definiuje szanse mutacji: 30% ADD, 40% MUTATE, 30% DELETE

- **`insert_passing_siding`** (1220-1387)
  - Wstawia mijankę ze zwrotnicami
  - Szuka 6+ prostych pod rząd
  - Zastępuje po 2 proste zwrotnicami (SL+SR) + wstawia gałąź (R-S-R lub L-S-L)
  - Obecnie tylko 1 prosta w gałęzi

- **`get_available_piece_types`** (1389-1419)
  - Liczy ile sztuk użyto dla każdego rodzaju elementu
  - Sprawdza co jeszcze zostało w magazynie
  - W Etapie 1 używanie zwrotnic zostało wyłączone

- **`mutate_add`** (1421-1485)
  - Dodaje nowy element do toru
  - Najpierw mamy 60% szans: element umieszczony blisko końca
  - Następnie jest 70% szans: dodaje zakręt, nie prostą
  - Nie wstawia elementów między zwrotnice

- **`mutate_change`** (1487-1535)
  - Zmienia typ elementu (np. prosta → zakręt)
  - Gdy złe zamknięcie: skupia zmianę elementu na końcowej 1/3 toru
  - NIGDY nie zmienia zwrotnic

- **`mutate_delete`** (1537-1605)
  - Usuwa zbędny element
  - Gdy tor za duży: usuwa proste
  - Gdy blisko granicy: usuwa elementy przy krawędzi
  - NIGDY nie usuwa zwrotnic (muszą być sparowane)

- **`mutate_shrink_for_boundary`** (1607-1647)
  - Kurczy tor gdy wychodzi poza granice
  - Usuwa proste jedna po drugiej aż się zmieści

- **`mutate_swap`** (1649-1671)
  - Zamienia miejscami dwa elementy
  - Elementy nie są zwrotnicami

- **`mutate_for_vertical_spread`** (1673-1751)
  - Przestawia tor żeby wykorzystać wysokość (oś Y)
  - Przenosi proste z poziomych odcinków na pionowe
  - Dzieli 8 zakrętów na 4+proste+4

- **`mutate_for_closure`** (1753-1807)
  - Naprawia złe zamknięcie pętli
  - Liczy kąty i sprawdza czego brakuje (L czy R zakręt)
  - Zamienia zakręty (L↔R) albo dodaje na koniec

- **`_can_change_to`** (1809-1821)
  - Sprawdza czy magazyn pozwala na zmianę typu
  - Pomocnicza funkcja dla mutate_change

- **`repair_boundary_violations`** (1823-1859)
  - Wywoływana po każdej mutacji
  - Usuwa proste aż tor zmieści się w granicach

- **`mutate`** (1861-1920) - **GŁÓWNA FUNKCJA**
  - Decyduje KTÓRĄ mutację zastosować
  - **Priorytety:**
    1. Granice (gdy przekroczone → 70% shrink, 30% inna mutacja)
    2. Gdy Etap 2: wstaw mijankę (30% szans)
    3. Złe zamknięcie (50% closure [inteligentna naprawa], 50% standardowa mutacja)
    4. Za mało pionu (30% vertical spread)
    5. Standardowo: losuj ADD/MUTATE/DELETE
  - ZAWSZE wywołuje repair_boundary_violations na koniec

- **`apply_multiple_mutations`** (1922-1932) *(Obecnie nieużywany kod ale zostawię w opisie)*
  - Tworzy kilka wariantów zmutowanych (domyślnie 3)
  - Zwraca listę - najlepszy zostanie wybrany później

#### 2. IDEA (15-20% nieosiągalnych w populacji)

**Kod:** linia 2085 (infeasible_ratio), linie 2675-2691 (utrzymywanie proporcji)

**Wartości:** 15% w Etapie 1 (linia 2901), 20% w Etapie 2 (linia 3014)

Etap 2 używa większego współczynnika (20%) ponieważ zwrotnice częściej tworzą nieosiągalne rozwiązania - gałąź musi się zamknąć między dwoma zwrotnicami, co jest trudniejszym ograniczeniem.

#### 3. DEB'S CONSTRAINT HANDLING (dwupoziomowa fitness)

**Kod:** linie 1091-1117 (obliczanie fitness)

Osobniki osiągalne ZAWSZE wygrywają z nieosiągalnymi (fitness > 0 vs < 0).

---

### Architektura Danych

**Hierarchia** (od atomu do populacji):

```python
Gene (linie 299-307)
    piece_type: PieceType       # STRAIGHT/CURVE_LEFT/CURVE_RIGHT/SWITCH_*
    is_branch: bool             # czy element w gałęzi zwrotnicy
    is_reversed: bool           # czy zwrotnica zamykająca

Individual (linie 891-909)
    genes: List[Gene]           # UPORZĄDKOWANA lista (kolejność = tor)
    fitness_result: FitnessResult
    age: int                    # wiek dla elityzmu

Population (linie 2070-2717)
    feasible: List[Individual]   # osobniki spełniające ograniczenia
    infeasible: List[Individual] # naruszające (15-20% dla IDEA)
```

**GENOTYP vs FENOTYP:**
- **Gene** = genotyp (To co budujemy: typ elementu)
- **TrackPiece** (linie 275-296) = fenotyp (To jest: obliczona geometria)
- **Transformacja:** `build_track_from_genes()` (linie 532-623) - forward propagation

---

## Część 2: Inicjalizacja i Selekcja

Metoda `Population.initialize()` (linie 2095-2173) tworzy początkową populację przez **MIESZANĄ STRATEGIĘ: 51% heurystyczna, 49% losowa**.

### Skąd te wartości?

- Kod jawnie tworzy heurystyczne wzorce w krokach 1-4 (linie 2106-2142)
- Krok 5 wypełnia resztę losowymi (linia 2168: `while len(...) < size`)
- Dla populacji 100: kroki 1-4 tworzą 15+15+16+5=51, reszta to 49 losowych
- Proporcje zmieniają się dla różnych rozmiarów populacji (np. dla 50: ~30/20)

### Konkretne kroki (dla population_size=100)

#### KROK 1: Max-piece patterns (15 osobników)

**Kod:** `min(size//6, 15)` → 15
**Funkcja:** `_create_max_piece_pattern()` (linie 2211-2284)

Oblicza ile prostych zmieści się w granicach (linie 2249-2264):
```python
extra_width = (boundary_width - 20) - 80  # margines - bazowe koło
max_straights = int(extra_width / 16) * 2  # 16 studs na prostą, 2 strony
```

Buduje symetryczny owal: `8R + S...S + 8R + S...S` (linie 2267-2282)

#### KROK 2: Vertical patterns (15 osobników)

**Kod:** `min(size//4, 15)` → 15
**Funkcja:** `_create_vertical_pattern()` (linie 2400-2476)

Proste domyślnie idą w prawo (kąt 0). Aby szły pionowo:
- 4 zakręty obraca kierunek o 90 stopni (4 × 22.5 = 90)
- Potem proste idą pionowo
- 8 zakrętów zawraca (180 stopni)
- Proste z powrotem pionowo

**Pattern:** `4R + S(pion) + 8R + S(pion) + 4R = 16 zakrętów = zamknięcie`

#### KROK 3: Varied patterns (16 osobników)

**Kod:** `size//6` → 16
**Funkcja:** `_create_varied_pattern()` (linie 2343-2398)

Losuje z 5 gotowych wzorów (owale, prostokąty, koła)

#### KROK 4: Warunkowy (5 osobników)

**Kod:** `if-elif-else` (linie 2122-2142), `min(5, size//6)` → 5

Kod zawsze wykonuje dokładnie 1 z 3 bloków:

**A. if max_same_dir >= 16** (np. 20 prawych LUB 18 lewych):
- Warunek: masz ≥ 16 zakrętów tego samego typu (lewe LUB prawe)
- → `_create_simple_loop()` (linie 2286-2341)
- → Czyste koło: 16 zakrętów jednego typu
- → SILNA GWARANCJA: 16 × 22.5 = 360 stopni (matematyka)
- Przykład: 20R + 8L → używa 16R do koła

**B. elif total_curves >= 16** (np. 12L + 10R = 22):
- Warunek: NIE spełnia A (max < 16) ALE suma ≥ 16
- → `_create_oval_pattern()` (linie 2175-2209)
- → Używa wszystkich dominujących + proste
- Przykład: 12R + 8L → używa wszystkie 12R + proste

**C. else:**
- Warunek: NIE spełnia A ani B (suma < 16)
- → `_create_varied_pattern()` (dodatkowe wzory)
- Przykład: 8R + 6L = 14 → losowy wzór

#### KROK 5: Random (49 osobników)

**Kod:** `while len(feasible+infeasible) < 100` (linie 2168-2171)
**Funkcja:** `_create_random()` (linie 2552-2575)

Losowa długość (12-24), losowe typy z dostępnego inwentarza

---

### Mechanizm Selekcji

#### 1. Selekcja Turniejowa (linie 2645-2650)

```python
tournament_size = 3  # Losuj 3 osobników
parent = max(tournament, key=lambda x: (
    x.fitness_result.is_feasible,    # Priorytet 1: osiągalni > nieosiągalni
    x.fitness_result.fitness          # Priorytet 2: większa fitness
))
```

**Eksperymentalnie:**
- Rozmiar 2 = za słaba presja selekcyjna (za losowo)
- Rozmiar 5+ = za silna presja (utrata różnorodności, "supergen" dominuje)
- Rozmiar 3 = kompromis

#### 2. Elityzm (linie 2617-2628)

```python
elite_count = max(2, len(feasible) // 10)  # 10% najlepszych, min 2

sorted_feasible = sorted(feasible, key=fitness, reverse=True)
for ind in sorted_feasible[:elite_count]:
    ind_copy = ind.copy()
    ind_copy.age += 1  # Starzenie
    new_feasible.append(ind_copy)  # BEZ mutacji
```

**Działa to tak:**
- Bierzemy tylko z populacji feasible (osiągalnych)
- Sortujemy od najlepszego do najgorszego (reverse=True)
- Wybieramy TOP 10% (ale minimum 2)
- Kopiujemy ich BEZ MUTACJI do nowej generacji
- Dla populacji z 85 feasible: elite_count = 85//10 = 8 chronionych

#### 3. Funkcja Fitness (linie 1091-1117)

**DLA OSIĄGALNYCH (fitness > 0):**
```python
weighted_pieces = suma(count * waga)  # zwrotnice liczą się x2
closure_quality = (1 - pos_error/tol) * 1000
switch_bonus = ile_zwrotnic * 500

fitness = weighted_pieces * 1000 + closure_quality + switch_bonus
# Typowo: 20000-50000
```

**DLA NIEOSIĄGALNYCH (fitness < 0):**
```python
constraint_violation = (
    pos_error * 100 +
    angle_error * 10 +
    boundary_violations * 1000 +
    collision_count * 500 +
    inventory_violation * 10000
)

fitness = -constraint_violation + weighted_pieces * 10
```

---

## Część 3: Filozofia Adaptacyjnych Mutacji

Funkcja `mutate()` (MutationOperator) nie losuje ślepo - **analizuje stan toru i wybiera mutację dopasowaną do problemu**.

### Hierarchia Priorytetów

Priorytety ułożone od **NAJTWARDSZYCH** do **NAJŁAGODNIEJSZYCH** ograniczeń:

#### 1. GRANICE
**Funkcja:** `mutate_shrink_for_boundary()`

To twardy warunek - tor musi się zmieścić w granicach. Nie ma sensu dodawać zwrotnic czy poprawiać zamknięcia jeśli tor i tak wychodzi na zewnątrz.

70% a nie 100% dlatego bo 30% czasu dajemy szansę na kreatywne rozwiązanie - może inna mutacja znajdzie lepszy sposób (np. zamieni proste na zakręty).

#### 2. ZWROTNICE (30% szans w Etapie 2)
**Funkcja:** `insert_passing_siding()`

Mijanka wymaga przeciwnych typów zwrotnic (SL+SR), bo gałąź PRZEKRACZA z jednej strony na drugą.

#### 3. ZAMKNIĘCIE PĘTLI
**Funkcja:** `mutate_for_closure()`

PODSTAWOWY CEL - tor musi się zamknąć w pętlę. Jeśli już mieści się w granicach (priorytet 1), możemy się tym zająć.

#### 4. WYKORZYSTANIE PIONU (30% gdy y_spread < 0.5)
**Funkcja:** `mutate_for_vertical_spread()`

OPTYMALIZACJA - tor już działa, ale nie jest najbardziej optymalny. Lepiej żeby był bardziej owalny niż płaski.

#### 5. STANDARDOWE MUTACJE (gdy brak powyższych problemów)
**Funkcje:** `mutate_add` (30%), `mutate_change` (40%), `mutate_delete` (30%)

Gdy tor nie ma oczywistych problemów, losuj standardowe mutacje dla eksploracji przestrzeni rozwiązań.

---

### Funkcje Geometryczne

**`build_track_from_genes()`** - buduje tor metodą FORWARD PROPAGATION
- Każdy element zaczyna tam gdzie poprzedni się skończył
- Pozycja N = f(pozycja N-1, kąt N-1, typ N)

**`calculate_closure_error()`** - sprawdza czy tor się zamyka
- Porównuje początek z końcem: błąd pozycji + błąd kąta
- Tolerancje: 8 studs, 15 stopni

**`check_boundary_violations()`** - czy tor wychodzi poza granice
- Dla zakrętów: próbkuje 5 punktów wzdłuż łuku (nie tylko początek/koniec)

**`check_collisions()`** - czy elementy się nie nakładają
- Min odległość: 6 studs
- Wyjątki: zamknięcie pętli, elementy przy zwrotnicach

**`validate_branch_closure()`** - czy gałąź mijanki się zamyka

---

## Część 4: Etapy

### Dlaczego dwa etapy?

Zwrotnice komplikują problem:
- Wymagają zamkniętych gałęzi (branch musi łączyć dwie zwrotnice)
- Gałąź tworzy drugą pętlę

**Rozwiązanie: podział na 2**
- **Etap 1:** Rozwiązanie prostszego problemu (proste + zakręty)
- **Etap 2:** Dodanie zwrotnic

---

### ETAP 1: Podstawowy Tor (linie 2849-2950)

**Cel:** Stworzyć zamkniętą pętlę używając max prostych + zakrętów

#### KROK 1: Analiza wykonalności matematycznej (linie 2859-2867)

`check_inventory_feasibility()` (linie 2747-2799):

**Warunki zamknięcia:**
- Prawych ≥ 16, LUB
- Lewych ≥ 16, LUB
- |Lewe - Prawe| ≥ 16

Jeżeli BRAK: `"WARNING: This inventory CANNOT form a closed loop!"`

#### KROK 2: Utworzenie inwentarza bez zwrotnic (linie 2887-2894)

```python
phase1_inventory = PieceInventory(
    straight=self.inventory.straight,
    curve_left=self.inventory.curve_left,
    curve_right=self.inventory.curve_right,
    switch_left=0,       # WYŁĄCZONE
    switch_right=0       # WYŁĄCZONE
)
```

#### KROK 3: Inicjalizacja populacji (linia 2896-2903)

```python
population = Population(
    size=100,
    inventory=phase1_inventory,  # Bez zwrotnic
    phase=1,                     # Znacznik etapu
    infeasible_ratio=0.15        # 15% nieosiągalnych
)
```

#### KROK 4: Ewolucja (linie 2910-2938)

```python
max_generations = 200
early_stop = 50  # Przerwij po 50 generacjach bez poprawy

for gen in range(200):
    population.evolve()
    if stagnation >= 50:
        break
```

#### KROK 5: Zapisanie najlepszego (linia 2940)

```python
self.phase1_result = population.get_best()
```

---

### ETAP 2: Dodawanie Zwrotnic (linie 2952-3112)

**Cel:** Dodać mijankę do toru z Etapu 1

#### KROK 1: Analiza wykonalności zwrotnic (linie 2971-2996)

`analyze_switch_feasibility()` sprawdza:
- `passing_siding_left`: SL + SR + wystarczająco zakrętów na R-S-R
- `passing_siding_right`: SR + SL + wystarczająco zakrętów na L-S-L
- `crossover`: SL + SR bezpośrednio połączone

Jeżeli `recommendation == 'none_feasible'`:
```python
return phase1_result  # Zwróć bez zwrotnic
```

#### KROK 2: Seedowanie populacji wynikiem Etapu 1 (linie 3008-3047)

```python
population = Population(
    size=100,
    inventory=self.inventory,  # PEŁNY inwentarz (ze zwrotnicami!)
    phase=2,                   # Znacznik etapu
    infeasible_ratio=0.2       # 20% nieosiągalnych (więcej!)
)

# 25% populacji = kopia wyniku Etapu 1
for _ in range(25):
    ind = phase1_result.copy()
    population._add_to_population(ind)

# 75% populacji = wersje z wstawionymi zwrotnicami
for _ in range(75):
    ind = phase1_result.copy()
    ind = mutator.insert_passing_siding(ind)  # Próba wstawienia
    population._add_to_population(ind)
```

#### KROK 3: Ewolucja z agresywniejszymi parametrami (linie 3066-3094)

```python
max_generations = 100  # Połowa z Etapu 1
early_stop = 30        # Szybsze przerwanie
```

#### KROK 4: Porównanie wyników (linie 3098-3112)

```python
# Zwróć LEPSZY na podstawie FITNESS (nie liczby elementów!)
if phase2_result.fitness > phase1_result.fitness:
    return phase2_result
else:
    return phase1_result  # Zwrotnice nie poprawiły
```

Znacznik `self.phase` kontroluje dostęp do operatora wstawiania zwrotnic.

---

## Podsumowanie

### Najważniejsze Mechanizmy

1. **Mutation-only + IDEA + Deb's constraint** = skuteczne radzenie sobie z restrykcyjnymi ograniczeniami geometrycznymi

2. **Seedowanie 51% heurystykami** (vertical patterns, max-piece) daje dobry punkt startowy zamiast czystego losu

3. **Adaptacyjne mutacje** (priorytet: granice → zwrotnice → zamknięcie) skupiają wysiłek tam gdzie problem

4. **Dwuetapowa optymalizacja** = dekompozycja złożonego problemu na prostszy (Etap 1) + dodanie zwrotnic (Etap 2)

### Liczby

- **Populacja:** 100 (51 heurystyki, 49 los)
- **Etap 1:** max 200 generacji, stop po 50 stagnacji
- **Etap 2:** max 100 generacji, stop po 30 stagnacji
- **Turniej:** 3 osobniki
- **Elityzm:** 10% (min 2)
- **IDEA:** 15% (Etap 1) / 20% (Etap 2) nieosiągalnych

---

**KONIEC**

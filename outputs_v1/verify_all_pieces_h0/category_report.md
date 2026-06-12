# Category report

## switch
- no feasible solution containing this element was seen
- best infeasible: util 77.1%, CV=183.17
- decoder drop reasons (final-population sample):
  - junction[0] (pos 0): siding geometry invalid (branch endpoint mismatch)
  - junction[1] (pos 12): siding geometry invalid (branch endpoint mismatch)
- context: a passing siding adds a parallel track segment inside the loop's existing bounding box

## cross
- no feasible solution containing this element was seen
- context: a CROSS_90 needs the loop to cross itself perpendicular; the figure-8 family spends >=24 R40 curves on two turning-circle lobes

## dc
- no feasible solution containing this element was seen
- context: a DOUBLE_CROSSOVER joins two parallel tracks 16 studs apart and both traversals must jointly cover all 4 ports

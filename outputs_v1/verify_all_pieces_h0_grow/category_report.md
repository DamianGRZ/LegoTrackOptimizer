# Category report

## switch
- no feasible solution containing this element was seen
- best infeasible: util 77.6%, CV=600.15
- decoder drop reasons (final-population sample):
  - junction[2] (pos 102): no OUT position at the required main distance
  - junction[0] (pos 97): no OUT position at the required main distance
  - junction[1] (pos 97): no OUT position at the required main distance
- context: a passing siding adds a parallel track segment inside the loop's existing bounding box

## cross
- no feasible solution containing this element was seen
- context: a CROSS_90 needs the loop to cross itself perpendicular; the figure-8 family spends >=24 R40 curves on two turning-circle lobes

## dc
- no feasible solution containing this element was seen
- context: a DOUBLE_CROSSOVER joins two parallel tracks 16 studs apart and both traversals must jointly cover all 4 ports

# Category report

## switch
- utilization: 39.0% (22.4pp below global best)
- pieces: 78, speed: 0.95 m/s, switch count: 2
- bbox: 448 x 80 studs in 500 x 500 box
- context: a passing siding adds a parallel track segment inside the loop's existing bounding box

## cross
- utilization: 56.7% (4.8pp below global best)
- pieces: 118, speed: 0.99 m/s, cross count: 1
- bbox: 496 x 496 studs in 500 x 500 box
- context: a CROSS_90 needs the loop to cross itself perpendicular; the figure-8 family spends >=24 R40 curves on two turning-circle lobes

## dc
- utilization: 61.4% (0.0pp below global best)
- pieces: 128, speed: 0.98 m/s, dc count: 1
- bbox: 480 x 176 studs in 500 x 500 box
- context: a DOUBLE_CROSSOVER joins two parallel tracks 16 studs apart and both traversals must jointly cover all 4 ports

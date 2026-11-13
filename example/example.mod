# Overview 01 example. Choose at most one option and pay a fixed cost
# if any option is chosen.

set OPTIONS;
param benefit {OPTIONS} >= 0;
param activation_cost >= 0;

var x {OPTIONS} binary;
var y;

maximize net_benefit:
    sum {i in OPTIONS} benefit[i] * x[i] - activation_cost * y;

subject to choose_at_most_one:
    sum {i in OPTIONS} x[i] <= 1;

# Since at most one binary x[i] can be 1, these links can be replaced
# by y >= sum {i in OPTIONS} x[i].
subject to activate_if_chosen {i in OPTIONS}:
    y >= x[i];

data;

set OPTIONS := A B C D E F G H I;
param benefit := A 9 B 12 C 11 D 5 E 8 F 4 G 10 H 7 I 6;
param activation_cost := 6;

# Expected optimum: 6, with x['B'] = 1 and y = 1.

set OPTIONS;
param benefit {OPTIONS} >= 0;
param activation_cost >= 0;

var x {OPTIONS} binary;
var y;

maximize net_benefit:
    sum {i in OPTIONS} benefit[i] * x[i] - activation_cost * y;

subject to choose_at_most_one:
    sum {i in OPTIONS} x[i] <= 1;

subject to activation_link:
    y >= sum {i in OPTIONS} x[i];

data;

set OPTIONS := A B C D E F G H I;
param benefit := A 9 B 12 C 11 D 5 E 8 F 4 G 10 H 7 I 6;
param activation_cost := 6;
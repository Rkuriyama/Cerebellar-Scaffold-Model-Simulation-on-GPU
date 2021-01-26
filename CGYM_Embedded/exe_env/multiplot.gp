TRIAL=ARG1

set term qt 1 size 1000, 1000
set multiplot

set ytics nomirror
set y2tics nomirror

set xzeroaxis

set lmargin screen 0.1
set rmargin screen 0.9
set tmargin screen 0.975
set bmargin screen 0.775

set yrange [-0.5:0.5]
set y2range [-0.1: 0.1]

plot "state_activity_data.res" every :::TRIAL::TRIAL  u 1:6 w l title "activiy0 push left", "state_activity_data.res" every :::TRIAL::TRIAL u 1:7 w l title "activity1 push right", "state_activity_data.res" every :::TRIAL::TRIAL u 1:($7-$6) w l axis x1y2 title "acivity diff (right - left)", "dg.res" every :::TRIAL::TRIAL u 2 w l axis x1y2 title "dg0", "dg.res" every :::TRIAL::TRIAL u 4 w l axis x1y2 title "dg1"

set lmargin screen 0.1
set rmargin screen 0.9
set tmargin screen 0.725
set bmargin screen 0.525

set yrange [-2.5: 2.5]
set y2range [-3: 3]
plot "state_activity_data.res" every :::TRIAL::TRIAL  u 1:2 w l axis x1y1 title "Cart Position", "state_activity_data.res" every :::TRIAL::TRIAL u 1:3 w l axis x1y2 title "Cart Velocity"

set lmargin screen 0.1
set rmargin screen 0.9
set tmargin screen 0.475
set bmargin screen 0.275

set yrange [-0.3: 0.3]
set y2range [-2: 2]
plot "state_activity_data.res" every :::TRIAL::TRIAL  u 1:4 w l axis x1y1 title "Pole Ang", "state_activity_data.res" every :::TRIAL::TRIAL u 1:5 w l axis x1y2 title "Pole Ang Velocity"

set lmargin screen 0.1
set rmargin screen 0.9
set tmargin screen 0.225
set bmargin screen 0.025

unset yrange
unset y2range
plot "freq_data.res" every :::TRIAL::TRIAL  u 1:2 w l title "Cart V 0", "freq_data.res" every :::TRIAL::TRIAL u 1:3 w l  title "Cart V 1", "freq_data.res" every :::TRIAL::TRIAL u 1:4 w l  title "Pole Ang 0", "freq_data.res" every :::TRIAL::TRIAL u 1:5 w l  title "Pole Ang 1", "freq_data.res" every :::TRIAL::TRIAL u 1:6 w l  title "Pole Ang V 0", "freq_data.res" every :::TRIAL::TRIAL u 1:7 w l  title "Pole Ang V 1" 


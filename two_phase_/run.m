
c = [70 130];
A = [12 6; 0 15; 2 8; 0 1];
b = [600; 300; 220; 10];
sense = ["<=";"<=";"<=";">="];

out = two_phase_simplex(c,A,b,sense);
disp(out.x); disp(out.z);

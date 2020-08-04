\* Source Pyomo model name=unknown *\

max 
x5:
+370 x1
+420 x2
-50 x3
+790 x4

s.t.

c_u_x6_:
+1 x1
+1 x2
+1 x3
+1 x4
<= 4

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= 1
   0 <= x2 <= 1
   0 <= x3 <= 1
   0 <= x4 <= 1
binary
  x1
  x2
  x3
  x4
end

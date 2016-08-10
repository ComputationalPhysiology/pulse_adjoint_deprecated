Rin   = 0.0;
R0 = 0.5;
R1  = 1.5;
R2 = 2.5;
R3 = 3.5;
psize = 0.25;

Point(1) = { Rin,  0.0, 0.0, psize };
Point(2) = { R1, 0.0, 0.0, psize };
Point(3) = { R2, 0.0, 0.0, psize };
Point(4) = { R3, 0.0, 0.0, psize };

Point(5) = { R0, 0.0, 0.0, psize };
Point(6) = { -R0, 0.0, 0.0, psize };

Line(1) = { 5, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };

Line(4) = { 1, 5 };

line_id = 3;
For num In {0:5}
    out[] = Extrude
        { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, Pi/3 }
        { Line{line_id}; };
    line_id = out[0];
    surf[num]    = out[1];
    bc_epi[num]  = out[2];
    bc_endo[num] = out[3];
EndFor



line_id = 2;
For num In {6:11}
    out[] = Extrude
        { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, Pi/3 }
        { Line{line_id}; };
    line_id = out[0];
    surf[num]    = out[1];
    bc_epi[num]  = out[2];
    bc_endo[num] = out[3];
EndFor

line_id = 1;
For num In {12:15}
    out[] = Extrude
        { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, Pi/2 }
        { Line{line_id}; };
    line_id = out[0];
    surf[num]    = out[1];
    bc_epi[num]  = out[2];
    bc_endo[num] = out[3];
EndFor

Circle(69) = {5, 1, 6};
Circle(70) = {5, 1, 22};
Circle(71) = {5, 1, 24};
Circle(72) = {24, 1, 6};
Circle(73) = {6, 1, 22};
Line Loop(74) = {66, 54, 58, 62};
Plane Surface(75) = {74};



Physical Surface("REGION_1") = {8};
Physical Surface("REGION_2") = {12};
Physical Surface("REGION_3") = {16};
Physical Surface("REGION_4") = {20};
Physical Surface("REGION_5") = {24};
Physical Surface("REGION_6") = {28};
Physical Surface("REGION_7") = {32};
Physical Surface("REGION_8") = {36};
Physical Surface("REGION_9") = {40};
Physical Surface("REGION_10") = {44};
Physical Surface("REGION_11") = {48};
Physical Surface("REGION_12") = {52};
Physical Surface("REGION_13") = {56};
Physical Surface("REGION_14") = {60};
Physical Surface("REGION_15") = {64};
Physical Surface("REGION_16") = {68};
Physical Surface("REGION_17") = {75};

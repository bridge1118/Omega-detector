function C = covrot(x, y, th)
% Create rotated covariance matrix.
% C = covrot(x, y, th)
% x, y are standard deviations and th is rotation angle
% $Id: gmmb_demo01.m,v 1.2 2005/04/14 10:33:34 paalanen Exp $

O = [x 0; 0 y];
R = [cos(th) -sin(th); sin(th) cos(th)];
M = R * O;
C = M * M';

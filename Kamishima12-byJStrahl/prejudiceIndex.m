function [NPI,PI] = prejudiceIndex(y,s)
J = [sum(~y & ~s),sum(~y & s);sum(y & ~s),sum(y & s)]/size(y,1);
PI = sum(sum(J.*log2(J./(sum(J,2)*sum(J,1))))); 
p_s = sum(s) / length(s);
H_b_s = -p_s * log2(p_s) - (1-p_s) * log2(1-p_s);
p_y_tst = sum(y) / length(y);
H_b_y_tst = -p_y_tst * log2(p_y_tst) - (1-p_y_tst) * log2(1-p_y_tst);
NPI = PI / sqrt(H_b_y_tst * H_b_s);
end
function [NMI, MI] = mutualInformationXY(y,y_hat)
J = [sum(~y_hat & ~y),sum(~y_hat & y);sum(y_hat & ~y),sum(y_hat & y)]/length(y);
MI = sum(sum(J.*log2(J./(sum(J,2)*sum(J,1))))); 
p_y_hat = sum(y_hat) / length(y_hat);
H_b_y_hat = -p_y_hat * log2(p_y_hat) - (1-p_y_hat) * log2(1-p_y_hat);
p_y_tst = sum(y) / length(y);
H_b_y_tst = -p_y_tst * log2(p_y_tst) - (1-p_y_tst) * log2(1-p_y_tst);
NMI = MI / sqrt(H_b_y_hat*H_b_y_tst);
end
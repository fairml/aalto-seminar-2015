function [UEI] = underEstimationIndex(y_hat,y,s)
p_hat_y_s = [sum(y_hat & s),sum(y_hat & ~s),sum(~y_hat & s),sum(~y_hat & ~s)]/length(y_hat);
p_samp_y_s = [sum(y & s),sum(y & ~s),sum(~y & s),sum(~y & ~s)]/length(y);
UEI = sqrt(1-(sum(sqrt(p_hat_y_s.*p_samp_y_s))));
end
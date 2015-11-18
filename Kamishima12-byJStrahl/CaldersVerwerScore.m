function [CVS] = CaldersVerwerScore(y,s)
p_y_s1 = sum(y & s) / sum(s);
p_y_s0 = sum(y & ~s) / sum(~s);
CVS = p_y_s1 - p_y_s0;
end

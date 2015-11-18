function [] = plot_functions(Male_, Female_, x_m, x_f, cdf_m, cdf_f, f_m, f_f)

%plot cumulative distribution functions (empirical and theoretical);
figure;
cdfplot(Male_) %cdf for males (empirical)
hold on;
cdfplot(Female_) %cdf for females (empirical)
plot(x_m,cdf_m,'LineWidth',2) %cdf for males (theoretical)
plot(x_f,cdf_f,'LineWidth',2) %cdf for females (theoretical)
legend('Male', 'Female');
title('cumulative distribution function');
hold off;
grid on

%plot probability density functions
title('PDF');
figure;
plot(x_m,f_m,'r');
axis([200 800 0 inf]);
hold on;
plot(x_f,f_f,'b');
hold off;
legend('Male', 'Female')
grid on
end
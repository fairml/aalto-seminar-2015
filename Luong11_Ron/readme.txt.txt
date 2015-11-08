Discrimination discovery algorithm of Luong et al. (2011) applied to the German credit score and the Adult census data set.

The script uses relative paths, therefore run it from the bin directory and retain the same folder structure (bin, data, results).

Usage: discovery.py [-cf] [-cm] [-anw] save_diff

	Please provide at least one flag for the analysis:
                    -cf: German Credit, Female Non-single
                    -cm: German Credit, Male, Married, 30 < Age < 60
                    -anw: Adult Census, Non-white person
        save_diff - Save plots and diff files (Boolean, 1 - saves to results folder, 0 - only shows plots, mandatory)
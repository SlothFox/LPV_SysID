# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:37:02 2021

@author: alexa
"""

import seaborn as sns

sns.set_theme(style="whitegrid")

tips = sns.load_dataset("tips")

ax = sns.boxplot(x=tips["total_bill"])

ax = sns.boxplot(x="day", y="total_bill", hue="smoker",

                 data=tips, palette="Set3")
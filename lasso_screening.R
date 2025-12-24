{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Preliminary feature screening using LASSO\
# Raw data are not provided\
\
library(glmnet)\
\
# internal_data <- read_excel("Internal.xlsx")\
\
y <- internal_data$VTE\
X <- model.matrix(VTE ~ ., data = internal_data)[, -1]\
\
set.seed(123)\
lasso_cv <- cv.glmnet(\
  X, y,\
  alpha = 1,\
  family = "binomial",\
  nfolds = 10\
)\
\
coef_1se <- coef(lasso_cv, s = "lambda.1se")\
selected_vars <- rownames(coef_1se)[coef_1se[,1] != 0][-1]\
\
print(selected_vars)\
}
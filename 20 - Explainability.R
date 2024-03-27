library(tidyverse)
library(tidymodels)
# install.packages('vip')
# install.packages('DALEX')
library(vip)
library(DALEX)



# Data and Model Setup  ---------------------------------------------------

set.seed(42)
cars <- read_csv('https://www.dropbox.com/scl/fi/xavej23qpauvx3xfdq7zh/car_sales.csv?rlkey=4mfp6tpia0uqkcoiqf9jleau3&dl=1') %>% 
  slice_sample(prop = .2)

cars_split <- initial_split(cars, strata = sellingprice_log)

cars_training <- cars_split %>% training()
cars_testing <- cars_split %>% testing()

cars_rec <- recipe(sellingprice_log ~ .,
                   data = cars_training) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(make, threshold = 0.03) %>% 
  step_other(model, threshold = 0.01) %>% 
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors())

# in case you want to look at the effect of the recipe:
cars_rec %>% prep() %>% juice() %>% glimpse()

lr_spec <- linear_reg()

xgb_spec <- boost_tree() %>% 
  set_engine('xgboost') %>% 
  set_mode('regression')

wkfl_xgb <- workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(cars_rec)

wkfl_lr <- workflow() %>% 
  add_model(lr_spec) %>% 
  add_recipe(cars_rec)

# Now run some models:
final_lr <- wkfl_lr %>% 
  last_fit(split = cars_split)

final_xgb <- wkfl_xgb %>% 
  last_fit(split = cars_split)



# Let's look at some explainability: --------------------------------------

top_features <- final_lr %>% extract_fit_parsnip() %>% tidy() %>%  dplyr::slice(2:11) %>% pull(term)


# Why does this produce "Warnings"? See also the next question.

# The errors are a results of the data not going through the recipe to be cleaned. As a result, the data does not match what the
# expects. The prediction column is not made into a factor also.

explainer_lr <- DALEX::explain(model = final_lr %>% extract_fit_parsnip(), 
                               data = cars_training %>% select(-sellingprice_log),
                               y = cars_training %>% pull(sellingprice_log), 
                               label = "Linear Regression")

# Those warnings lead to an error here. Why?
# Hint: You can also compare the DALEX::explain() function call above to the one below.

# There are no dummy coded variables so the model isn't getting the same columns as it was trained on

model_profile(explainer_lr, variables = top_features)



# Here is one method for generating "explainers":
explainer_lr <- DALEX::explain(model = final_lr %>% extract_fit_parsnip(), 
                               data = cars_rec %>% prep() %>% juice() %>% select(-sellingprice_log),
                               y = cars_training %>% pull(sellingprice_log), 
                               label = "Linear Regression")

explainer_xgb <- DALEX::explain(model = final_xgb %>% extract_fit_parsnip(), 
                                data = cars_rec %>% prep() %>% juice() %>% select(-sellingprice_log),
                                y = cars_training %>% pull(sellingprice_log), 
                                label = "XGBoost")


# Compare performance of the two models:
final_lr %>% collect_metrics()
final_xgb %>% collect_metrics()

# Please go here to learn about partial dependence plots:
# https://chat.openai.com/share/f391c97a-3133-497c-a853-5227901d1699

# Now see if you can interpret these plots:
# Why do they look so different? 

# The linear regression produces linear results while the XG Boost makes decisions in a different way.
#They both say that the same features have a big effect on the model

# What does this tell you about why xgboost has better performance than linear regression?

#The xg boost has slope that varies for the features depending on the x axis. This demonstrates that prices fluxuate
# at non-constant rates. For example the first 25,000 miles on a car affect cost much more than the next 50,000

pdp_lr <- model_profile(explainer_lr, variables = top_features)
pdp_xgb <- model_profile(explainer_xgb, variables = top_features)
plot(pdp_lr)
plot(pdp_xgb)


# And another method:
# Why do these variable importance plots look so different?
# Again, what does that tell you about the fundamental differences between xgboost and linear regression?

# XG Boost handles multicollinearity using feature selection.
# These plots look different because of the way that the models handle each column. Fundamentally, a linear regression
# fits everything to be a line for better or worse. XG boost accurately reflects the weight of a step on the X
# axis for a feature based on how high or low it is.


vip(final_lr %>% extract_fit_parsnip())
vip(final_xgb %>% extract_fit_parsnip())






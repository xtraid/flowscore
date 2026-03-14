df= read.csv("../data/profiles.csv")

#importo libreria per fare vif
library(car)

df$working_category <- as.factor(df$working_category)

#media dei salari
df$mean_income <- sapply(df$monthly_income_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  mean(nums)
})

#sd salari
df$sd_income <- sapply(df$monthly_income_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  sd(nums)
})

#percentuale varianza
df$percentage_sd <-df$sd_income/df$mean_income

#trend salari
df$income_trend <- sapply(df$monthly_income_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  lm(nums ~ seq_along(nums))$coefficients[2]
})

#media risparmi
df$mean_saving <- sapply(df$monthly_saving_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  mean(nums)
})
df$percentage_saving <- df$mean_saving/df$mean_income
#media spese fisse
df$mean_fixed_exp <- sapply(df$monthly_fixed_exp_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  mean(nums)
})

#media spese variabili
df$mean_variable_exp <- sapply(df$monthly_variable_exp_hist, function(x) {
  nums <- as.numeric(strsplit(gsub("\\[|\\]", "", x), ",")[[1]])
  mean(nums)
})

#booleano dei default
df$bool_defaults <- ifelse(df$int_defaults >= 1, 1, 0)

#percentuale spese fisse
df$percentage_fixed_exp<- df$mean_fixed_exp/df$mean_income

#percentuale esposizione su bnpl
df$percentage_debt <- df$bnpl_exposure/df$mean_income

#liquidità su spese, gestire se non ho risparmi
df$liquidity_buffer <- ifelse(df$mean_saving == 0, NA, df$mean_fixed_exp/df$mean_saving)


#calcolo il modello con variabile risposta il default
model <- glm(bool_defaults ~ mean_income + percentage_sd + working_category + percentage_fixed_exp + percentage_debt
             + pay_on_time_bills  + income_trend + percentage_saving,
             data = df,
             family = binomial)
summary(model)
anova(model)

#vedo la correlazione
vif(model)
#calcolo flowscore
df$flowscore <- (1- predict(model, type = "response"))*100

View(df)
write.csv(df[, c("id", "flowscore")], "scores_output.csv", row.names = FALSE)

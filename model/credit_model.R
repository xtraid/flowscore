df= read.csv("../data/profiles.csv")

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

#vedo la correlazione
cor(df[, c("mean_income",
           "sd_income",
           "percentage_fixed_exp",
           "percentage_debt",
           "pay_on_time_bills")])

#calcolo il modello con variabile risposta il default
model <- glm(bool_defaults ~ mean_income + sd_income + working_category + percentage_fixed_exp + percentage_debt
             + pay_on_time_bills,
             data = df,
             family = binomial)
summary(model)
anova(model)

#calcolo flowscore
df$flowscore <- (1- predict(model, type = "response"))*100

View(df)

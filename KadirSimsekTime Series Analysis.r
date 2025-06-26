#Time Series Project Kadir Şimşek-2502375

devtools::install_github("twitter/AnomalyDetection")
#install.packages("tseries")

library(tseries)
library(ggplot2)
library(forecast)
library(devtools)
library(AnomalyDetection)
library(tseries)
library(uroot)
library(anomalize)
library(zoo)
library(readr)
library(dplyr)
library(forecast)
library(TSA)
library(fpp2)
library(readxl)
library(tidyverse)

#install.packages("fpp2")
#1 introduction covering data description
data <- read_csv("PNRGINDEXM.csv")
str(data)
ts_data <- ts(data$PNRGINDEXM, start = c(1992, 01), end= c(2024,10),frequency = 12)
ts_data
#Time Series plot
autoplot(ts_data)+
  ggtitle("Global Price of Energy Index")+
  xlab("Date")+
  ylab("Energy Price")+
  theme_minimal()

#acf and pacf
library(gridExtra)
g1<-ggAcf(ts_data,lag.max = 48)+theme_minimal()+ggtitle("ACF of Data")
g2<-ggPacf(ts_data,lag.max = 48)+theme_minimal()+ggtitle("PACF of Data")
grid.arrange(g1,g2,ncol=2)

#split test and train dataset
test_data <- tail(ts_data,12)
test_data

train_data <- head(ts_data, -12)
train_data

acf(train_data)
pacf(train_data)

#detect anomaly
test_data_raw<- tail(data,12)
train_data_raw <-head(data, -12)

train_data_raw %>% 
  anomalize::time_decompose(PNRGINDEXM, method="stl", frequency = 12, trend = "auto") %>%
  anomalize::anomalize(remainder, method="gesd", alpha=.05, max_anoms=0.1) %>%
  anomalize::plot_anomaly_decomposition()


#Extracting and Cleaning the Anomalous Data Points
train_data_cleaned <- train_data_raw %>%
  anomalize::time_decompose(PNRGINDEXM) %>%
  anomalize::anomalize(remainder) %>%
  anomalize::time_recompose() %>%
  clean_anomalies()

View(train_data_cleaned)
head(train_data_cleaned,1)


clean_train_data<- ts(train_data_cleaned$observed,start = c(1992,1),end = c(2023,10),frequency = 12)
acf(clean_train_data)
pacf(clean_train_data)


#after detecting and removing anomaly points we continue the analysis


#box-cox transformation on our train data
lambda <- BoxCox.lambda(clean_train_data) #
lambda

#optimal lambda is close to 0 so we can apply log transformation, or we can apply boxcox transformation.
traindata_bc <- BoxCox(clean_train_data,lambda)

#check whether the transformation works or not
lambda1<-BoxCox.lambda(traindata_bc)
lambda1

#We can see that the transformation works.

autoplot(traindata_bc)+theme_minimal()

#Firstly, let’s decide which type of trend exists in the series. We can use KPSS test.
kpss.test(traindata_bc,null="Level") #The series is not stationary.

kpss.test(traindata_bc,null="Trend")#The series has stochastic trend

adf.test(traindata_bc)


library(pdR)
test<-HEGY.test(traindata_bc, itsd=c(0,0,0))
test$stats

ndiffs(traindata_bc)
nsdiffs(traindata_bc)
#To make the series stationary, one regular differencing will be enough.

#Firstly, let's take regular difference
autoplot(diff(traindata_bc))

test2<-HEGY.test(diff(traindata_bc), itsd=c(0,0,0))
test2$stats
kpss.test(diff(traindata_bc),null="Level") #The series is  stationary.
kpss.test(diff(traindata_bc),null="Trend")# The series is deterministic.

adf.test(diff(traindata_bc))#Reject H0 so there are not any unit root. So series is stationary.

library(gridExtra)
p1<-ggAcf(diff(traindata_bc),lag.max = 72)
p2<-ggPacf(diff(traindata_bc),lag.max = 72)
grid.arrange(p1,p2,ncol=2)

eacf(diff(traindata_bc))

#arima(1,1,0)
#arima(1,1,1)
#arima(2,1,1)
#arima(0,1,1)
#arima(0,1,0)
#sarima(0,1,1)(1,0,1)(12)
#sarima(1,1,1)(1,0,1)(12)
#sarima(1,1,1)(0,0,1)(12)
#sarima(1,1,1)(1,0,0)(12)

#choosing best model 

fit1 <- Arima(traindata_bc, order= c(1,1,0))
fit1

fit2 <- Arima(traindata_bc, order= c(1,1,1))
fit2

fit3 <- Arima(traindata_bc, order= c(2,1,1))
fit3

fit4 <- Arima(traindata_bc, order= c(0,1,1))
fit4

fit5 <-Arima(traindata_bc,order = c(0, 1, 1), seasonal = c(1, 0, 1))
fit5

fit6 <-Arima(traindata_bc,order = c(1, 1, 1), seasonal = c(1, 0, 1))
fit6

fit7 <-Arima(traindata_bc,order = c(1, 1, 1), seasonal = c(0, 0, 1))
fit7

fit8 <-Arima(traindata_bc,order = c(1, 1, 1), seasonal = c(1, 0, 0))
fit8

fit9<-Arima(traindata_bc,order = c(1, 1, 0), seasonal = c(1, 0, 1))
fit9

#fit9 best model


#diagnostics checks
r=resid(fit9)

autoplot(r)+geom_line(y=0)+theme_minimal()+ggtitle("Plot of The Residuals")

#qq-plot
ggplot(r, aes(sample = r)) +stat_qq()+geom_qq_line()+ggtitle("QQ Plot of the Residuals")+theme_minimal()

ggplot(r,aes(x=r))+geom_histogram(bins=20)+geom_density()+ggtitle("Histogram of Residuals")+theme_minimal()

summary(r)

ggplot(r,aes(y=r,x=as.factor(1)))+geom_boxplot()+ggtitle("Box Plot of Residuals")+theme_minimal()

library(tseries)
jarque.bera.test(r)
shapiro.test(r)

#Since p value is less than alpha , we reject Ho. Therefore,it can be said that we have no enough evidence to claim that we have residuals with normal distribution

ggAcf(as.vector(r),main="ACF of the Residuals",lag = 48)+theme_minimal() #to see time lags, as. factor function is used.

#If all spikes are in the WN band, the residuals are uncorrelated. In the ACF, almost all spikes are in the WN band. To be sure, let us apply formal tests.

library(TSA) #for zlag function
m = lm(r ~ 1+zlag(r))

library(lmtest)
bgtest(m,order=15) #order is up to you

#Since p value is greater than α, we have 95% confident that the residuals of the model are uncorrelated, according to results of Breusch-Godfrey Test.

Box.test(r,lag=15,type = c("Ljung-Box"))

#Since p value is greater than α, we have 95% confident that the residuals of the model are uncorrelated, according to results of Box-Ljung Test.

Box.test(r,lag=15,type = c("Box-Pierce"))

#Since p value is greater than α, we have 95% confident that the residuals of the model are uncorrelated, according to results of Box-Pierce Test.

rr=r^2
g3<-ggAcf(as.vector(rr), lag.max = 48)+theme_minimal()+ggtitle("ACF of Squared Residuals")
g4<-ggPacf(as.vector(rr), lag.max = 48)+theme_minimal()+ggtitle("PACF of Squared Residuals")  # homoscedasticity check
grid.arrange(g3,g4,ncol=2)


library(lmtest)
m = lm(r ~ traindata_bc+zlag(traindata_bc)+zlag(traindata_bc,2))
bptest(m)


m1 = lm(r ~ traindata_bc+zlag(traindata_bc)+zlag(traindata_bc,2)+zlag(traindata_bc)^2+zlag(traindata_bc,2)^2+zlag(traindata_bc)*zlag(traindata_bc,2))
bptest(m1)

#Since p value is smaller than α, we fail reject Ho. Therefore, I can say that we have enough evidence to claim that there is heteroscedasticity problem, according to results of White test.





#install.packages("FinTS")
library(MTS)
archTest(rr)


library(rugarch)
spec_garch <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(1, 0))) 
fit_garch <- ugarchfit(spec = spec_garch, data = clean_train_data)
print(fit_garch)

# sGARCH(2,2)
spec_sgarch <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(2, 2)),
                          mean.model = list(armaOrder = c(1, 0)))
fit_sgarch <- ugarchfit(spec = spec_sgarch, data = clean_train_data)
print(fit_sgarch)

# apARCH(2,2)
spec_aparch <- ugarchspec(variance.model = list(model = "apARCH", garchOrder = c(2, 2)),
                          mean.model = list(armaOrder = c(1, 0)))
fit_aparch <- ugarchfit(spec = spec_aparch, data = clean_train_data)
print(fit_aparch)


f_arma <- forecast(fit9,h=12)
f_arma

autoplot(f_arma)+theme_minimal()+ggtitle("Forecast of SARIMA Model")

f_t <- InvBoxCox(f_arma$mean,lambda)
autoplot(f_t,main=c("Comparison of forecast vs actual test"), series="forecast" ) + autolayer(test_data,series = "actual") + theme_bw()


#
boot_1 <- ugarchboot(fit_garch,method=c("Partial","Full")[1],n.ahead = 12,n.bootpred=1000,n.bootfit=1000)

f_1 <- boot_1@forc@forecast$seriesFor
f_vec1 <- as.vector(f_1)
#

boot_2 <- ugarchboot(fit_sgarch,method=c("Partial","Full")[1],n.ahead = 12,n.bootpred=1000,n.bootfit=1000)

f_2 <- boot_2@forc@forecast$seriesFor
f_vec_2 <- as.vector(f_2)
#

boot_3 <- ugarchboot(fit_aparch,method=c("Partial","Full")[1],n.ahead = 12,n.bootpred=1000,n.bootfit=1000)

f_3 <- boot_3@forc@forecast$seriesFor
f_vec_3 <- as.vector(f_3)
#
#For SARIMA model
accuracy(f_t, test_data)

#For GARCH model
accuracy(f_vec1, test_data)

#For sGARCH(2,2) Model
accuracy(f_vec_2, test_data)

#For apARCH(2,2) model
accuracy(f_vec_3, test_data)

#sGARCH(2,2)

x11()


ts_garch <- ts(f_1, frequency = 12, start = c(2023, 11))
ts_garch_1 <- ts(f_2, frequency = 12, start = c(2023,11))
ts_aparch <- ts(f_3, frequency = 12, start = c(2023,11))

autoplot(f_t, series = "SARIMA (1,1,0)(1,0,1)", main = "Comparison of Forecast Values") + 
  autolayer(test_data, series = "actual") + 
  autolayer(ts_garch, series = "ARFIMA(1,0,1)+GARCH(1,1)") +
  autolayer(ts_garch_1, series = "ARFIMA(1,0,1)+sGARCH(2,2)") +
  autolayer(ts_aparch, series = "ARFIMA(1,0,1)+apARCH(2,2)") +theme_bw()

# Based on the analysis, sGARCH(2,2) is the most accurate model in terms of RMSE and MAPE.
# However, if capturing volatility is critical, apARCH(2,2) would be a better choice.
# For this project, sGARCH(2,2) is recommended for its balance between accuracy and simplicity.



#Forecast

#ETS
hosp.ets <- ets(clean_train_data, model = "ZZZ")
summary(hosp.ets)

#ETS(M,A,N) model is suggested by the ets() function. (Multiplicative error + Additive trend + no seasonality).

hosp.f2 <- forecast::forecast(hosp.ets, h = 12)
autoplot(hosp.f2)+autolayer(test_data,series="actual",color="red")

# ETS FORECAST

plot(ts_data, lwd = 2, main = "ETS")
lines(hosp.f2$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(hosp.f2$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(hosp.f2$lower[, 2], start = c(2023, 11), frequency = 12)
UI <- ts(hosp.f2$upper[, 2], start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)
#
autoplot(hosp.f2$mean, series = "ETS")+autolayer(test_data,series="actual")+theme_bw()

accuracy(hosp.f2,testdata)

#**TBATS**
tbatsmodel <- tbats(clean_train_data)
tbatsmodel

#
tbats_forecast <- forecast::forecast(tbatsmodel,h=12)
autoplot(tbats_forecast)+autolayer(test_data,series="actual",color="red")+theme_minimal()
#

plot(ts_data, lwd = 2, main = "TBATS")
lines(tbats_forecast$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(tbats_forecast$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(tbats_forecast$lower[, 2], start = c(2023, 11), frequency = 12)
UI <- ts(tbats_forecast$upper[, 2], start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)
#
autoplot(tbats_forecast$mean, series = "TBATS")+autolayer(test_data,series="actual")+theme_bw()

#TBATS accuracy
accuracy(tbats_forecast,test_data)


#**Neural Network

nnmodel<-nnetar(clean_train_data)
nnmodel

#Neural Network
nnforecast <- forecast::forecast(nnmodel,h=12,PI=TRUE)
accuracy(nnforecast,test_data)

autoplot(clean_train_data)+autolayer(fitted(nnmodel))+theme_minimal()+ggtitle("Fitted Values of NN Model")

autoplot(nnforecast)+theme_minimal()+autolayer(test_data,series="actual",color="red")


#
plot(ts_data, lwd = 2, main = "Neural Network")
lines(nnforecast$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(nnforecast$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(nnforecast$lower[, 2], start = c(2023, 11), frequency = 12)
UI <- ts(nnforecast$upper[, 2], start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)

#
autoplot(nnforecast$mean, series = "NN")+theme_bw() + autolayer(test_data, series = "actual")
#

#**Hyperparameter Tuning in Neural Network**
p <- c(1, 2, 3)
P <- c(1, 12, 24)
size <- c(128, 64, 32)
repeats <- c(10, 15, 20,40,80)

results <- data.frame(
  p = numeric(),
  P = numeric(),
  size = numeric(),
  repeats = numeric(),
  RMSE = numeric()
)

for (p in p) {
  for (P in P) {
    for (size in size)
      for(repeats in repeats) {{
        nnm <-nnetar(clean_train_data, p = p, P = P, size = size, repeats = repeats)
        nnf <- forecast(nnm, h = 12)
        accur <- accuracy(nnf,test_data)
        rmse <- accur[2,2]
        
        results <- rbind(results, data.frame(
        p = p, 
        P = P, 
        size = size,
        repeats = repeats,
        RMSE = rmse
      ))
      
    }}}}

best_params <- results[which.min(results$RMSE), ]
best_params
#

#hyper neural network
nnm <-nnetar(clean_train_data, p = 8, P = 12, size = 32, repeats = 80)
nnf <- forecast(nnm, h = 12, PI = TRUE)
accur_nn <- accuracy(nnf,test_data)
accur_nn

autoplot(nnf)+autolayer(test_data,series="actual",color="red")+theme_minimal()

#
plot(ts_data, lwd = 2, main = "Hyper Neural Network")
lines(nnf$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(nnf$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(nnf$lower[, 2], start = c(2023, 11), frequency = 12)
UI <- ts(nnf$upper[, 2], start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)
#

autoplot(nnf$mean, series = "NN")+theme_bw() + autolayer(test_data, series = "actual")








#**Prophet

library(prophet)
ds<-c(seq(as.Date("1992/01/01"),as.Date("2023/10/01"),by="month"))
df<-data.frame(ds,y=as.numeric(clean_train_data))
head(df)

#prophet
train_prophet <- prophet(df)
future <- make_future_dataframe(train_prophet,periods = 12)
forecast <- predict(train_prophet, future)
accuracy(tail(forecast$yhat,12),test_data)

plot(train_prophet, forecast)+theme_minimal()

prophet_plot_components(train_prophet, forecast)


prop_fit <- ts(head(forecast$yhat,298),start=c(1992,1), end=c(2023,10), frequency = 12)
prop <- ts(tail(forecast$yhat,12), start=c(2023,11), frequency = 12)

plot(ts_data, lwd = 2, main = "Prophet")
lines(prop_fit, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(prop, col = "blue", lty = 1, lwd = 2)
LI <- ts(tail(forecast$yhat_lower,12), start = c(2023, 11), frequency = 12)
UI <- ts(tail(forecast$yhat_upper,12), start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)


autoplot(prop, series = "Prophet Forecast")+autolayer(test_data,series="actual")+theme_minimal()


library(prophet)
library(forecast)

changepoint_prior <- c(0.1, 0.5, 0.9)
seasonality_prior <- c(0.1, 0.3, 0.5)
changepoint_range <- c(0.6, 0.8, 0.9)

results <- data.frame(
  changepoint_prior = numeric(),
  seasonality_prior = numeric(),
  changepoint_range = numeric(),
  RMSE = numeric()
)


for (cp in changepoint_prior) {
  for (sp in seasonality_prior) {
    for (cr in changepoint_range) {
      m <- prophet(
        changepoint.prior.scale = cp,
        seasonality.prior.scale = sp,
        changepoint.range = cr
      )
      m <- fit.prophet(m, df) 
      

      future <- make_future_dataframe(m, periods = 12, freq = "month")
      forecast <- predict(m, future)
      
      predicted <- tail(forecast$yhat, 12)
      acc <- accuracy(predicted, test_data)  
      rmse <- acc["Test set", "RMSE"]  # Extract RMSE from accuracy
      
      results <- rbind(results, data.frame(
        changepoint_prior = cp, 
        seasonality_prior = sp, 
        changepoint_range = cr, 
        RMSE = rmse
      ))
    }
  }
}


#best parameters
best_params1 <- results[which.min(results$RMSE), ]
best_params1


hosp_prophet_new <- prophet(df,changepoint.range=0.6,changepoint.prior.scale=0.1,seasonality.prior.scale=0.5)
future_new=make_future_dataframe(hosp_prophet_new,periods = 12, freq = "month")
forecast_new <- predict(hosp_prophet_new, future_new)

accuracy(tail(forecast_new$yhat,12),test_data)

plot(hosp_prophet_new, forecast_new)+theme_minimal()



prop_fit2 <- ts(head(forecast_new$yhat,298),start=c(1999,1), end=c(2023,10), frequency = 12)
prop2 <- ts(tail(forecast_new$yhat,12), start=c(2023,11), frequency = 12)

plot(ts_data, lwd = 2, main = "hyper prophet")
lines(prop_fit2, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+10/12, col = "red", lwd = 2)
lines(prop2, col = "blue", lty = 1, lwd = 2)
LI <- ts(tail(forecast_new$yhat_lower,12), start = c(2023, 11), frequency = 12)
UI <- ts(tail(forecast_new$yhat_upper,12), start = c(2023, 11), frequency = 12)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)



hype.prop <- ts(tail(forecast_new$yhat,12),start=c(2023,11), frequency = 12 )

autoplot(hype.prop, series = "Hyper Prophet Forecast")+autolayer(testdata,series="actual")+theme_minimal()

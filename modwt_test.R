library(waveslim)
library(ggplot2)
library(quantmod)
setwd("C:/Users/Nick/Desktop/TensorANFIS-master/TensorANFIS-master")


symbol <- "SPY"  
series <- new.env()
getSymbols.yahoo(symbol, env = series, from="2009-01-01", to=Sys.Date() + 1, auto.assign=T) 
stock <- series[[symbol]]
stock <- stock[, "SPY.Close"]

returns <- na.omit(diff(log(stock)))
volatility <- abs(returns)
plot.ts(returns)
plot.ts(volatility)

## Haar
haar <- modwt(returns, "la8", n.levels =  3, boundary = "periodic")
MRA <- mra(returns, "la8", method = 'modwt', 3, boundary = "periodic")


D1 <- haar[["d1"]]
D2 <- haar[["d2"]]
D3 <- haar[["d3"]]
S3 <- haar[["s3"]]

W1 <- MRA[["D1"]]
W2 <- MRA[["D2"]]
W3 <- MRA[["D3"]]
V3 <- MRA[["S3"]]

plot.ts(D1)
plot.ts(D2)
plot.ts(D3)
plot.ts(S3)

plot.ts(W1)
plot.ts(W2)
plot.ts(W3)
plot.ts(V3)

recon <- D1 + D2 + D3 + S3
recon2 <- W1 + W2 + W3 + V3

plot.ts(recon)
plot.ts(recon2)
plot.ts(stock)

x <- seq(1, length(recon), 1)
df <- data.frame(x,recon,recon2)

df1 <- data.frame(D1,D2,D3,S3)
df2 <- data.frame(W1,W2,W3,V3)

ggplot(df, aes(x)) +          
  geom_line(aes(y=recon), colour="red") +  
  geom_line(aes(y=recon2), colour="blue") 

write.csv(df1, file = 'modwt.csv')
write.csv(df2, file = 'modwt_mra.csv')

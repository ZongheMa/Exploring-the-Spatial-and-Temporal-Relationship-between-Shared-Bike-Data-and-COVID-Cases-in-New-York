# Load the required libraries
library(maptools)
library(lattice)
library(spdep)
library(sp)
library(rgdal)
library(tmap)
library(ggplot2)
library(gridExtra)
library(gstat)
library(OpenStreetMap)
library(spacetime)

#set working directory accordingly with setwd()
ca_counties <- readOGR(dsn="data/geo/Geography-resources/MODZCTA_2010.shp")
W <- nb2listw(poly2nb(ca_counties))
W
kable(listw2mat(W))

plot(ca_counties$collision_, ylab="Monthly average temperature", xlab="Time
(in months)", type="l")

counties_weekly_collision <- read.csv(file='counties_weekly_collision.csv')

plot(counties_weekly_collision$Alameda.County, ylab="Monthly average temperature", xlab="Time
(in months)", type="l")

counties_weekly_collision.mat <- as.matrix(counties_weekly_collision[2:59])
W <- listw2mat(W)
stacf(counties_weekly_collision.mat, W, 100)
counties_weekly_collision.mat.diff <- diff(counties_weekly_collision.mat,lag = 1, differences=1)
stacf(counties_weekly_collision.mat.diff, W, 80)

stpacf(counties_weekly_collision.mat,W,100)
stpacf(counties_weekly_collision.mat.diff,W,100)

W_fit<-list(w1=W)
fit.star <- starima_fit(counties_weekly_collision.mat[1:150,],W_fit,p=2,d=1,q=2)



pre.star <- starima_pre(counties_weekly_collision.mat[150:179,],
                        model=fit.star)


matplot(1:25,cbind(counties_weekly_collision.mat[150:174,1],pre.star$PRE[,1]),type="l")


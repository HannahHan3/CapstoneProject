clustring.types
dim(clustring)
is.na(clustring)
sum(is.na(clustring))
summary(clustring)
which(names(clustring)=="CBSA_CODE_-99999.0")
which(names(clustring)=="CBSA_CODE_49620.0")
which(names(clustring)=="CERTIFICATIONLEVEL_-9999.0")
which(names(clustring)=="CERTIFICATIONLEVEL_100.0")
which(names(clustring)=="LISTING_SOURCE_Co-List")
which(names(clustring)=="LISTING_SOURCE_MLS")
which(names(clustring)=="MARKET_ID_-9999.0")
which(names(clustring)=="MARKET_ID_9050.0")
which(names(clustring)=="MLSNAME_Columbus BOR")
which(names(clustring)=="MLSNAME_rs_gamls")
which(names(clustring)=="STATE_AL")
which(names(clustring)=="STATE_WI")
   
clustring.BATHROOM

cluster.X <- clustring[,-c(91:192, 226:228, 243:308, 325:350),drop=FALSE]
dim(cluster.X)

#cluster.X=clustring[,1:353]
set.seed(1234)
#km.out=kmeans(x=cluster.X,centers=8)
km.out=kmeans(x=cluster.X,centers=5,nstart=10)
km.out

wss <- (nrow(cluster.X)-1)*sum(apply(cluster.X,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(cluster.X,
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Sum of Squares (Objective)")

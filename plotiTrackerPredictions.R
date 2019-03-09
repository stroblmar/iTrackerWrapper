# ========================================================================
# Plot the predictions of iTracker as in the publication
# ========================================================================
# Load Libraries
# ========================================================================
rm(list=ls()) # Reset environment
library(ggplot2)
# ========================================================================
# Configure Script
# ========================================================================
# setwd("~/Dropbox/Projects/gaze_tracking_Project/code/iTrackerWrapper/")
gazePredictionsDf = read.csv("data/sampleVideo/gazePredictions.csv")
labelVec = as.numeric(gazePredictionsDf$EstPosY<(-6)) # True target for each frame (here: 0=eyes, 1=mouth). This one is fake, created simply for illustration.
referencePosDf = read.csv("referencePosDF_faces.csv") # Reference positions the participants were asked to look at
referencePosDf = referencePosDf[referencePosDf$Task=="enlargedFace",]
screenSize = c(-11.7,-0.9,-5.3,1.1) # Phone screen size: c(BottomY,TopY,LeftX,RightX)
figSize = c(15,18) # Size of the figure
rightEyePos = c(-1.2,-4) # Bottom left corner of right eye
leftEyePos = c(-5.1,-4) # Bottom left corner of left eye
mouthPos = c(-4,-8.7) # Bottom left corner of mouth
eyeH = 0.8
eyeW = 1.8
mouthH = 0.8
mouthW = 3.4
faceParamList = list(rightEyePos,leftEyePos,mouthPos,eyeH,eyeW,mouthH,mouthW)
outName = "examplePredictions.png" # Name of the saved figure
calibMethod = "raw" # Choose calibration method. Note because we have no training data in this demo only "raw" works. Otherwise options are: c("raw","fc1_model","preserveVar_model")
# ========================================================================
# Function to generate plots
# ========================================================================
PlotPredictions = function(estPosXVec,estPosYVec,labelVec,referencePosDf,screenSize,faceParamList,titleStr="",outName="",figSize=c(15,18),xlimVec=c(-7,2.5),ylimVec=c(-13,0)) {
  #,trueFocusVec,dataSetName,taskName,titleStr="",outName="") {
  # Assemble the data into a df for plotting
  dataDF = data.frame(X=c(referencePosDf$X,estPosXVec),
                      Y=c(referencePosDf$Y,estPosYVec),
                      Label=as.factor(c(c(0,0,1),labelVec)), # [eye, eye, mouth, custom labels]
                      Type=as.factor(c(rep(0,nrow(referencePosDf)),rep(1,length(estPosXVec))))) # 0=Reference Position, 1=Estimated Position

  # Plot
  # Draw the screen and the face outline
  rightEyePos = faceParamList[[1]] # Bottom left corner of right eye
  leftEyePos = faceParamList[[2]] # Bottom left corner of left eye
  mouthPos = faceParamList[[3]] # Bottom left corner of mouth
  eyeH = faceParamList[[4]]
  eyeW = faceParamList[[5]]
  mouthH = faceParamList[[6]]
  mouthW = faceParamList[[7]]

  p = ggplot(dataDF,aes(x=X,y=Y)) +
    annotate("rect", xmin=screenSize[3], xmax=screenSize[4], ymin=screenSize[1], ymax=screenSize[2],
             alpha = .2) +
    annotate("rect", xmin=leftEyePos[1], xmax=leftEyePos[1]+eyeW, ymin=leftEyePos[2], ymax=leftEyePos[2]+eyeH,
             alpha = .3) +
    annotate("rect", xmin=rightEyePos[1], xmax=rightEyePos[1]+eyeW, ymin=rightEyePos[2], ymax=rightEyePos[2]+eyeH,
             alpha = .3) +
    annotate("rect", xmin=mouthPos[1], xmax=mouthPos[1]+mouthW, ymin=mouthPos[2], ymax=mouthPos[2]+mouthH,
             alpha = .3)
  # Draw the rest
  p = p +
    coord_equal()+
    geom_point(aes(colour=Label,shape=Type),size=4,stroke=1.5) +
    scale_colour_manual(values=c("#d8b365","#5ab4ac"), #"#1F3B7A","#B6B61E"
                        name="True Focus",
                        labels = c("Eyes", "Mouth")) +
    scale_shape_manual(values=c(16,3)) +
    geom_point(data=dataDF[dataDF$Type==0,],aes(x=X,y=Y),shape=21,colour="black",size=4,stroke=1) +
    theme_bw() +
    xlab("Horizontal Position Relative to \n Camera in cm") +
    ylab("Vertical Position Relative to \n Camera in cm") +
    xlim(xlimVec) +
    ylim(ylimVec) +
    ggtitle(titleStr) +
    theme(legend.position="none",text = element_text(size=20),
          axis.text=element_text(size=rel(1)),
          axis.title=element_text(size=22))

  # Save if a filename has been provided
  if(outName!="") {
    ggsave(outName,p,width=figSize[1],height=figSize[2],units="cm")
  } else {
    return(p)
  }
}
# ========================================================================
# Main Body
# ========================================================================
# Perform post-processing calibration (if requested). Note for this to work
# you must have training data available (and for the SVR also pre-trained the
# model and saved it to disk).
if(calibMethod=="raw") {
  estPosXVec = gazePredictionsDf$EstPosX
  estPosYVec = gazePredictionsDf$EstPosY
  # ------------------------------------------------------------
} else if(calibMethod=="preserveVar_model") {
  # Assemble predictions and true positions into one data frame
  # meanCalibGrid = c(mean(gridPtDF$X),mean(gridPtDF$Y))
  # varCalibGrid = c(var(gridPtDF$X),var(gridPtDF$Y))
  # meanObserved = c(mean(calibrationDF$EstPosX),mean(calibrationDF$EstPosY))
  # varObserved = c(var(calibrationDF$EstPosX),var(calibrationDF$EstPosY))
  # # Estimate parameters for model
  # a = sqrt(varCalibGrid/varObserved)
  # b = meanCalibGrid - a*meanObserved
  # model_X = function(x) {a[1]*x + b[1]}
  # model_Y = function(y) {a[2]*y + b[2]}
  # # Adjust the predictions
  # estPosXVec = model_X(gazeEstimatesDF$EstPosX)
  # estPosYVec = model_Y(gazeEstimatesDF$EstPosY)
  # ------------------------------------------------------------
} else {
  # Load the calibration model
  # load(paste0(modelDir,calibMethod,"_modelX_",subjectName,".rda"))
  # load(paste0(modelDir,calibMethod,"_modelY_",subjectName,".rda"))
  # estPosXVec = predict(model_X, gazeEstimatesDF)
  # estPosYVec = predict(model_Y, gazeEstimatesDF)
}

# Plot results
PlotPredictions(estPosXVec,estPosYVec,
                labelVec=labelVec,referencePosDf=referencePosDf,
                screenSize=screenSize,faceParamList=faceParamList,
                outName=outName)
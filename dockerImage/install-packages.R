## Create the personal library if it doesn't exist. Ignore a warning if the directory already exists.
dir.create(Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
## Install one package.
install.packages("ggplot2", Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )
## Install a package that you have copied to the remote system.
## install.packages("file_name.tar.gz", Sys.getenv("R_LIBS_USER")
## Install multiple packages.
## install.packages(c("timeDate","robustbase"), Sys.getenv("R_LIBS_USER"), repos = "http://cran.case.edu" )
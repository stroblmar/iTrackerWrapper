import time
import platform
import multiprocessing as mp
# =============================================================
# Class to manage jobs during multithreading. Basically does queue management. Jobs can be submitted
# to a queue and the manager distributes them across the cores.
# I'm aware that this class is equivalent to multiprocessing.Pool(). However, multiprocessing.Pool() doesn't work
# with cafe because it wants to pickle objects when sending them to the workers and cafe objects don't
# like to be pickled...
class ThreadManager():
    def __init__(self, nCores=2):
        self.nCores = nCores
        self.toDoList = []
        self.activeProcessList = []
        self.finishedProcessList = [] # Jobs which have finished running
        self.completedJobList = [] # Jobs which have been joined

    # Function to return the number of running processes managed by the manager
    def NProcRunning(self):
        return len(self.activeProcessList)

    # Function to submit jobs
    def RunJob(self,newJobObj,verboseB=False):
        jobSubmittedB = False
        while not jobSubmittedB:
            nProcRunning = self.NProcRunning()
            if verboseB: print "Number of active cores: " + str(nProcRunning) + " of " + str(self.nCores)
            # If there is a free core submit the job
            if nProcRunning < self.nCores:
                if verboseB: print "Starting calculations for subject " + str(newJobObj)
                newJobObj.start()
                self.activeProcessList.append(newJobObj)
                jobSubmittedB = True
            else:  # Wait for a job to finish and then submit
                for job in self.activeProcessList:
                    # Check if the job has finished
                    if verboseB: print "Checking process " + str(job)
                    if job.is_alive():
                        if verboseB: print "Process " + str(job) + " hasn't finished."
                        job.join(timeout=1)
                        # job.exitcode()
                        # time.sleep(0.5)  # Not finished, so wait for a bit and then check the next
                    else:
                        if verboseB: print "Process " + str(job) + " finished. Submitting next job"
                        self.activeProcessList.remove(job)
                        self.finishedProcessList.append(job)
                        break
        return 0

    # Function to add a job to the queue
    def AddJob(self,jobObj):
        self.toDoList += [jobObj]

    # Function to process a queue of jobs
    def RunQueue(self,verboseLevel=1):
        # Run the jobs
        for i,job in enumerate(self.toDoList):
            if verboseLevel>=1: print " JobManager: -- Processing job "+str(i)+" of "+str(len(self.toDoList))
            self.RunJob(job,verboseLevel>=2)
        # Make sure all processes completed, and collect results
        allProcessList = self.finishedProcessList+self.activeProcessList
        self.completedJobList = self.JoinJobs(allProcessList,verboseLevel>=2)
        return self.completedJobList

    # Function to check processes are completed, and results can be retrieved
    def JoinJobs(self,jobList,verboseB=False):
        completedProcessList = []
        for job in jobList:
            if verboseB: print "Joining process " + str(job)
            job.join()
            completedProcessList.append(job)
        return completedProcessList

# =============================================================
# Function to print system info. Taken from http://sebastianraschka.com/Articles/2014_multiprocessing.html
# Useful for finding out how many cores one has etc.
def print_sysinfo():

    print('\nPython version  :', platform.python_version())
    print('compiler        :', platform.python_compiler())

    print('\nsystem     :', platform.system())
    print('release    :', platform.release())
    print('machine    :', platform.machine())
    print('processor  :', platform.processor())
    print('CPU count  :', mp.cpu_count())
    print('interpreter:', platform.architecture()[0])
    print('\n\n')
# Configuration file for jupyter
# ==============================================================================
c.NotebookApp.ip = '0.0.0.0' # Set to 0.0.0.0 so that is accessible from host. See https://stackoverflow.com/questions/38830610/access-jupyter-notebook-running-on-docker-container
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.MultiKernelManager.default_kernel_name = 'python2'
c.NotebookApp.allow_root = True # New security feature in jupyter doesn't like it to be run as root (which it is from inside docker). This is a workaround.

# Turn off the password and token features which would otherwise require me to enter a password/token every time.
c.NotebookApp.password_required = False
c.NotebookApp.token = ''
import os

wrapper = '#!/bin/bash\nsource /opt/anaconda3/etc/profile.d/conda.sh\nconda activate tap\nexec claude-tap "$@"\n'
path = os.path.expanduser('~/local/bin/claude-tap')
with open(path, 'w') as f:
    f.write(wrapper)
os.chmod(path, 0o755)
print('Wrapper created at', path)

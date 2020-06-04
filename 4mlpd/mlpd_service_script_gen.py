import os

cwd = os.getcwd()

script = """[Unit]
Description=Machine learning Python daemon (MLPD)
After=slurmctld.service
ConditionPathExists=""" + cwd + """/mlpd.conf

[Service]
Type=simple
WorkingDirectory=""" + cwd + """
ExecStart=/bin/python3 """ + cwd + """/main.py
PIDFile=/run/mlpd.pid
LimitNOFILE=65536
TasksMax=infinity"""

file = open('mlpd.service', 'w')
file.write(script)
file.close()

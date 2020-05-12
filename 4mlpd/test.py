import numpy as np
import os
import sys
import argparse
import datetime
import time
import pwd
import grp
import gettext
import configparser


import MySQLdb

#from pseudo_cluster.task import Task_record, TASK_STATES
#from pseudo_cluster.tasks_list import Tasks_list

slurmdbdconf_str = "[fake_header]\n" + open("/etc/slurm-llnl/slurmdbd.conf", 'r').read()
#print(slurmdbdconf_str)
slurmdbdconf = configparser.RawConfigParser()
slurmdbdconf.read_string(slurmdbdconf_str)
db_login = slurmdbdconf.get("fake_header", "StorageUser")
db_password = slurmdbdconf.get("fake_header", "StoragePass")
#db_host = slurmdbdconf.get("fake_header", "StorageHost")
db_host = "localhost"
#db_port = slurmdbdconf.get("fake_header", "StoragePort")
db_port = 3306
db_storage = slurmdbdconf.get("fake_header", "StorageLoc")

db = MySQLdb.connect(
    host=db_host,
    user=db_login,
    passwd=db_password,
    port=int(db_port),
    db=db_storage
)

cursor = db.cursor()

query = \
    """
        select *
        /*    id_job,
            job_name,
            time_submit,
            time_start,
            time_end,
            id_user,
            id_group,
            timelimit,
            cpus_req,
            mem_req*/
        from
            cluster_job_table
      """

cursor.execute(query)
print(cursor.fetchall())

import MySQLdb
import configparser
import gc

# +--------------------+---------------------+------+-----+------------+----------------+
# | Field              | Type                | Null | Key | Default    | Extra          |
# +--------------------+---------------------+------+-----+------------+----------------+
# | job_db_inx         | bigint(20) unsigned | NO   | PRI | NULL       | auto_increment |
# | mod_time           | bigint(20) unsigned | NO   |     | 0          |                |
# | deleted            | tinyint(4)          | NO   |     | 0          |                |
# | account            | tinytext            | YES  |     | NULL       |                |
# | admin_comment      | text                | YES  |     | NULL       |                |
# | array_task_str     | text                | YES  |     | NULL       |                |
# | array_max_tasks    | int(10) unsigned    | NO   |     | 0          |                |
# | array_task_pending | int(10) unsigned    | NO   |     | 0          |                |
# | cpus_req           | int(10) unsigned    | NO   |     | NULL       |                |
# | derived_ec         | int(10) unsigned    | NO   |     | 0          |                |
# | derived_es         | text                | YES  |     | NULL       |                |
# | exit_code          | int(10) unsigned    | NO   |     | 0          |                |
# | job_name           | tinytext            | NO   |     | NULL       |                |
# | id_assoc           | int(10) unsigned    | NO   | MUL | NULL       |                |
# | id_array_job       | int(10) unsigned    | NO   | MUL | 0          |                |
# | id_array_task      | int(10) unsigned    | NO   |     | 4294967294 |                |
# | id_block           | tinytext            | YES  |     | NULL       |                |
# | id_job             | int(10) unsigned    | NO   | MUL | NULL       |                |
# | id_qos             | int(10) unsigned    | NO   | MUL | 0          |                |
# | id_resv            | int(10) unsigned    | NO   | MUL | NULL       |                |
# | id_wckey           | int(10) unsigned    | NO   | MUL | NULL       |                |
# | id_user            | int(10) unsigned    | NO   | MUL | NULL       |                |
# | id_group           | int(10) unsigned    | NO   |     | NULL       |                |
# | pack_job_id        | int(10) unsigned    | NO   | MUL | NULL       |                |
# | pack_job_offset    | int(10) unsigned    | NO   |     | NULL       |                |
# | kill_requid        | int(11)             | NO   |     | -1         |                |
# | mcs_label          | tinytext            | YES  |     | ''         |                |
# | mem_req            | bigint(20) unsigned | NO   |     | 0          |                |
# | nodelist           | text                | YES  |     | NULL       |                |
# | nodes_alloc        | int(10) unsigned    | NO   | MUL | NULL       |                |
# | node_inx           | text                | YES  |     | NULL       |                |
# | partition          | tinytext            | NO   |     | NULL       |                |
# | priority           | int(10) unsigned    | NO   |     | NULL       |                |
# | state              | int(10) unsigned    | NO   |     | NULL       |                |
# | timelimit          | int(10) unsigned    | NO   |     | 0          |                |
# | time_submit        | bigint(20) unsigned | NO   |     | 0          |                |
# | time_eligible      | bigint(20) unsigned | NO   | MUL | 0          |                |
# | time_start         | bigint(20) unsigned | NO   |     | 0          |                |
# | time_end           | bigint(20) unsigned | NO   | MUL | 0          |                |
# | time_suspended     | bigint(20) unsigned | NO   |     | 0          |                |
# | gres_req           | text                | NO   |     | ''         |                |
# | gres_alloc         | text                | NO   |     | ''         |                |
# | gres_used          | text                | NO   |     | ''         |                |
# | wckey              | tinytext            | NO   |     | ''         |                |
# | work_dir           | text                | NO   |     | ''         |                |
# | system_comment     | text                | YES  |     | NULL       |                |
# | track_steps        | tinyint(4)          | NO   |     | NULL       |                |
# | tres_alloc         | text                | NO   |     | ''         |                |
# | tres_req           | text                | NO   |     | ''         |                |
# +--------------------+---------------------+------+-----+------------+----------------+


def slurm_db():
    # надо доделать обработку конфига
    slurmdbdconf_str = "[fake_header]\n" + open("/etc/slurm-llnl/slurmdbd.conf", 'r').read()
    # print(slurmdbdconf_str)
    slurmdbdconf = configparser.RawConfigParser()
    slurmdbdconf.read_string(slurmdbdconf_str)
    db_login = slurmdbdconf.get("fake_header", "StorageUser")
    db_password = slurmdbdconf.get("fake_header", "StoragePass")
    # db_host = slurmdbdconf.get("fake_header", "StorageHost")
    db_host = "localhost"
    # db_port = slurmdbdconf.get("fake_header", "StoragePort")
    db_port = 3306
    db_storage = slurmdbdconf.get("fake_header", "StorageLoc")

    db = MySQLdb.connect(
        host=db_host,
        user=db_login,
        passwd=db_password,
        port=int(db_port),
        db=db_storage
    )

    cursor = db.cursor(MySQLdb.cursors.DictCursor)
    query = "select * from cluster_job_table"
    cursor.execute(query)
    data = cursor.fetchall()

    cursor.close()
    db.close()
    gc.collect()

    return data

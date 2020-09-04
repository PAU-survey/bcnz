import os
import psycopg2

def connect_db():
    """Connect to the databased."""

    # Probably there is a more standardized way of doing this.
    pw_path = os.path.expanduser('~/paudm_pw_readonly')
    pw = open(pw_path).readline().strip()
    cred = {'database': 'dm',
            'user': 'readonly',
            'host': 'db.pau.pic.es',
            'password': pw}

    conn = psycopg2.connect(**cred)

    return conn

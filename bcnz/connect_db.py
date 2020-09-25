# Copyright (C) 2020 Martin B. Eriksen
# This file is part of BCNz <https://github.com/PAU-survey/bcnz>.
#
# BCNz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BCNz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BCNz.  If not, see <http://www.gnu.org/licenses/>.
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

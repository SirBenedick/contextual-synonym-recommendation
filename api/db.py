import logging
import logging.config
import sqlite3
import datetime

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: base Path on enviroment
PATH_TO_DATABASE_FOLDER = "/usr/share/sqlite"
# PATH_TO_DATABASE_FOLDER = "."  # overwrite path for local development

DATABASE_FEEDBACK_PATH = f"{PATH_TO_DATABASE_FOLDER}/feedback.db"
DATABASE_MONITORING_PATH = f"{PATH_TO_DATABASE_FOLDER}/monitoring.db"
DATABASE_FEEDBACK_TABLE_NAME = "tblFeedback"
DATABASE_MONITORING_REQUESTS_TABLE_NAME = "tblRequests"


def addFeedback(masked_sentence, original, replacement, score, rank_of_recommendation):
    try:
        # Setup sqlite connection
        sqliteConnection = sqlite3.connect(DATABASE_FEEDBACK_PATH)
        cursor = sqliteConnection.cursor()
        logger.debug("Connected to SQLite - feedback.db")

        # Check if table exists, if not create
        cursor.execute(
            f"""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{DATABASE_FEEDBACK_TABLE_NAME}' """
        )
        if cursor.fetchone()[0] == 0:
            logger.debug("Table does not exist.")
            sqlite_create_table_query = f"""CREATE TABLE {DATABASE_FEEDBACK_TABLE_NAME}(timestamp text,
                                                                    masked_sentence text, 
                                                                    original text, 
                                                                    replacement text, 
                                                                    score integer,
                                                                    rank_of_recommendation integer);"""

            cursor.execute(sqlite_create_table_query)
            logger.debug("Table created")

        # insert feedback detail
        sqlite_insert_with_param = f"""INSERT INTO '{DATABASE_FEEDBACK_TABLE_NAME}'
                                        ('timestamp', 'masked_sentence', 'original', 'replacement', 'score', 'rank_of_recommendation') 
                                        VALUES (?, ?, ?, ?, ?, ?);"""

        data_tuple = (
            datetime.datetime.now(),
            masked_sentence,
            original,
            replacement,
            score,
            rank_of_recommendation,
        )
        cursor.execute(sqlite_insert_with_param, data_tuple)
        sqliteConnection.commit()
        logger.debug("Feedback added successfully \n")

        cursor.close()
    except sqlite3.Error as error:
        logger.debug("Error while working with SQLite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            logger.debug("sqlite connection is closed")


def getAllFeedback(only_yesterday: bool = False) -> list:
    where_clause = ""
    if only_yesterday:
        where_clause = """WHERE DATE("NOW") > DATE(timestamp) and DATE("NOW","-2 days") < DATE(timestamp)"""
    sqlite_select_query = f"""SELECT masked_sentence, original, replacement, score, rank_of_recommendation, timestamp FROM {DATABASE_FEEDBACK_TABLE_NAME} {where_clause}"""
    result = executeQueryOnDB(
        sqlite_query=sqlite_select_query, path_to_db_file=DATABASE_FEEDBACK_PATH
    )
    return result


def logRequest(masked_sentence, original_token, recommended_synonyms, time_in_s):
    logger.debug(f"Masked sentence: {masked_sentence}")
    logger.debug(f"Original token: {original_token}")
    logger.debug(f"Recommended synonyms: {recommended_synonyms}")
    logger.debug(f"Time needed for prediction: {time_in_s}s")
    try:
        # Setup sqlite connection
        sqliteConnection = sqlite3.connect(DATABASE_MONITORING_PATH)
        cursor = sqliteConnection.cursor()
        logger.debug("Connected to SQLite - monitoring.db")

        # Check if table exists, if not create
        cursor.execute(
            f""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{DATABASE_MONITORING_REQUESTS_TABLE_NAME}' """
        )
        if cursor.fetchone()[0] == 0:
            logger.debug(
                f"Table '{DATABASE_MONITORING_REQUESTS_TABLE_NAME}' does not exist."
            )
            sqlite_create_table_query = f"""CREATE TABLE {DATABASE_MONITORING_REQUESTS_TABLE_NAME}(timestamp text,
                                                                    masked_sentence text, 
                                                                    original_token text, 
                                                                    recommended_synonyms blob, 
                                                                    time_in_s real);"""

            cursor.execute(sqlite_create_table_query)
            logger.debug("Table created")

        # insert feedback detail
        sqlite_insert_with_param = f"""INSERT INTO '{DATABASE_MONITORING_REQUESTS_TABLE_NAME}'
                                        ('timestamp', 'masked_sentence', 'original_token', 'recommended_synonyms', 'time_in_s') 
                                        VALUES (?, ?, ?, ?, ?);"""

        data_tuple = (
            datetime.datetime.now(),
            masked_sentence,
            original_token,
            str(recommended_synonyms),
            time_in_s,
        )
        cursor.execute(sqlite_insert_with_param, data_tuple)
        sqliteConnection.commit()
        logger.debug("Request added successfully \n")

        cursor.close()
    except sqlite3.Error as error:
        logger.debug("Error while working with SQLite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            logger.debug("sqlite connection is closed")


def getAllRequestsMonitoring(only_yesterday: bool = False) -> list:
    where_clause = ""
    if only_yesterday:
        where_clause = """WHERE DATE("NOW") > DATE(timestamp) and DATE("NOW","-2 days") < DATE(timestamp)"""
    sqlite_select_query = f"""SELECT masked_sentence, original_token, recommended_synonyms, time_in_s, timestamp FROM {DATABASE_MONITORING_REQUESTS_TABLE_NAME} {where_clause}"""
    result = executeQueryOnDB(
        sqlite_query=sqlite_select_query, path_to_db_file=DATABASE_MONITORING_PATH
    )

    return result


def get_overview_db():
    try:
        overview = {}
        # Get number of total requests per day
        exec = executeQueryOnDB(
            sqlite_query=f"""SELECT date (timestamp) date, COUNT(timestamp) FROM {DATABASE_MONITORING_REQUESTS_TABLE_NAME} GROUP BY date (timestamp);""",
            path_to_db_file=DATABASE_MONITORING_PATH,
        )
        for row in exec:
            if row[0] not in overview:
                overview[row[0]] = {}
            overview[row[0]]["requests"] = {"total": row[1]}

        # Get max, min and average of recommendation duration for each day
        exec = executeQueryOnDB(
            sqlite_query=f"""SELECT date (timestamp) date, COUNT(timestamp), max(time_in_s), min(time_in_s), avg(time_in_s) FROM {DATABASE_MONITORING_REQUESTS_TABLE_NAME} GROUP BY date (timestamp);""",
            path_to_db_file=DATABASE_MONITORING_PATH,
        )
        for row in exec:
            overview[row[0]]["requests"]["time_for_recommendation_in_s"] = {
                "maximum": row[2],
                "minimum": row[3],
                "average": row[4],
            }

        # Get total number of feedback for each day
        exec = executeQueryOnDB(
            sqlite_query=f"""SELECT date (timestamp) date, COUNT(timestamp) FROM {DATABASE_FEEDBACK_TABLE_NAME} GROUP BY date (timestamp)""",
            path_to_db_file=DATABASE_FEEDBACK_PATH,
        )

        for row in exec:
            if row[0] not in overview:
                overview[row[0]] = {}
            overview[row[0]]["feedback"] = {"total": row[1]}

        for date in overview:
            exec = executeQueryOnDB(
                sqlite_query=f"""SELECT score, COUNT(timestamp) FROM {DATABASE_FEEDBACK_TABLE_NAME} WHERE rank_of_recommendation < 5 AND date(timestamp) = '{date}' GROUP BY score ORDER BY score ASC;""",
                path_to_db_file=DATABASE_FEEDBACK_PATH,
            )
            if exec:
                good = 0
                bad = 0
                for row in exec:
                    if row[0] == 0:
                        bad = row[1]
                    if row[0] == 1:
                        good = row[1]

                overview[date]["feedback"]["rank_top_five"] = {
                    "total": bad + good,
                    "good": good,
                    "bad": bad,
                }

    except:
        return False

    # Convert overview dict into a sorted list
    overview_keys = list(overview.keys())
    overview_keys.sort(reverse=True)

    overview_list = []

    for key in overview_keys:
        # Add date as a field to the original object
        overview[key]["date"] = key
        overview_list.append(overview[key])

    return overview_list


def executeQueryOnDB(sqlite_query, path_to_db_file):
    try:
        # Setup sqlite connection
        sqliteConnection = sqlite3.connect(path_to_db_file)
        cursor = sqliteConnection.cursor()
        logger.debug("Connected to SQLite - monitoring.db")

        result: list = []
        for row in cursor.execute(sqlite_query):
            result.append(list(row))

        cursor.close()

        return result

    except sqlite3.Error as error:
        logger.debug("Error while working with SQLite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            logger.debug("sqlite connection is closed")

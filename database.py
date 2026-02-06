import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional, Tuple


DB_PATH = "museum_flow.db"


@contextmanager
def get_conn(db_path: str = DB_PATH):
  conn = sqlite3.connect(db_path, check_same_thread=False)
  try:
    yield conn
  finally:
    conn.close()


def init_db(db_path: str = DB_PATH):
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    # 检查表是否存在，如果存在但缺少新字段，则添加
    cur.execute(
      """
      SELECT name FROM sqlite_master WHERE type='table' AND name='raw_events';
      """
    )
    table_exists = cur.fetchone() is not None
    
    if not table_exists:
      # 创建新表
      cur.execute(
        """
        CREATE TABLE raw_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          timestamp REAL NOT NULL,
          person_id INTEGER NOT NULL,
          x REAL NOT NULL,
          y REAL NOT NULL,
          roi_id INTEGER,
          heading REAL,
          cluster_id INTEGER,
          frame_id INTEGER
        );
        """
      )
    else:
      # 检查并添加新字段（如果不存在）
      cur.execute("PRAGMA table_info(raw_events);")
      columns = [col[1] for col in cur.fetchall()]
      if "timestamp" not in columns:
        cur.execute("ALTER TABLE raw_events ADD COLUMN timestamp REAL;")
        # 为现有数据填充 timestamp（从 ts 字段解析）
        cur.execute("UPDATE raw_events SET timestamp = julianday(ts) * 86400.0 WHERE timestamp IS NULL;")
      if "frame_id" not in columns:
        cur.execute("ALTER TABLE raw_events ADD COLUMN frame_id INTEGER;")
    cur.execute(
      """
      CREATE TABLE IF NOT EXISTS stay_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        roi_id INTEGER NOT NULL,
        enter_time TEXT NOT NULL,
        leave_time TEXT NOT NULL,
        total_time REAL NOT NULL
      );
      """
    )
    cur.execute(
      """
      CREATE TABLE IF NOT EXISTS daily_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        roi_id INTEGER NOT NULL,
        total_visitors INTEGER NOT NULL,
        avg_dwell_time REAL NOT NULL
      );
      """
    )
    conn.commit()


def insert_raw_event(
  ts: datetime,
  person_id: int,
  x: float,
  y: float,
  roi_id: Optional[int],
  heading: Optional[float],
  cluster_id: Optional[int],
  frame_id: Optional[int] = None,
  db_path: str = DB_PATH,
):
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    timestamp = ts.timestamp()
    cur.execute(
      """
      INSERT INTO raw_events (ts, timestamp, person_id, x, y, roi_id, heading, cluster_id, frame_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
      """,
      (ts.isoformat(), timestamp, person_id, x, y, roi_id, heading, cluster_id, frame_id),
    )
    conn.commit()


def insert_stay_record(
  person_id: int,
  roi_id: int,
  enter_time: datetime,
  leave_time: datetime,
  total_time: float,
  db_path: str = DB_PATH,
):
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    cur.execute(
      """
      INSERT INTO stay_records (person_id, roi_id, enter_time, leave_time, total_time)
      VALUES (?, ?, ?, ?, ?);
      """,
      (person_id, roi_id, enter_time.isoformat(), leave_time.isoformat(), total_time),
    )
    conn.commit()


def aggregate_daily_stats(date_str: str, db_path: str = DB_PATH):
  """对指定日期做 GROUP BY 聚合，写入 daily_stats。

  date_str 形如 '2026-02-06'
  """
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    cur.execute(
      """
      SELECT roi_id,
             COUNT(DISTINCT person_id) AS visitors,
             AVG(total_time) AS avg_dwell
      FROM stay_records
      WHERE date(enter_time) = ?
      GROUP BY roi_id;
      """,
      (date_str,),
    )
    rows: Iterable[Tuple[int, int, float]] = cur.fetchall()

    for roi_id, visitors, avg_dwell in rows:
      cur.execute(
        """
        INSERT INTO daily_stats (date, roi_id, total_visitors, avg_dwell_time)
        VALUES (?, ?, ?, ?);
        """,
        (date_str, roi_id, visitors, avg_dwell),
      )

    conn.commit()


def get_today_flow_by_time(db_path: str = DB_PATH):
  """简单按时间统计当日馆内人数曲线，用于历史趋势图。"""
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    cur.execute(
      """
      SELECT strftime('%H:%M', ts) as t, COUNT(DISTINCT person_id)
      FROM raw_events
      WHERE date(ts) = date('now', 'localtime')
      GROUP BY t
      ORDER BY t;
      """
    )
    rows = cur.fetchall()
  times = [r[0] for r in rows]
  counts = [r[1] for r in rows]
  return times, counts


def get_historical_events(
  start_time: datetime,
  end_time: datetime,
  db_path: str = DB_PATH,
):
  """查询指定时间段内的历史事件数据，用于视频回溯。"""
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    cur.execute(
      """
      SELECT ts, timestamp, person_id, x, y, roi_id, heading, cluster_id, frame_id
      FROM raw_events
      WHERE timestamp >= ? AND timestamp <= ?
      ORDER BY timestamp, person_id;
      """,
      (start_time.timestamp(), end_time.timestamp()),
    )
    rows = cur.fetchall()
  return rows


def get_heatmap_data(
  minutes: int = 5,
  db_path: str = DB_PATH,
):
  """获取过去 N 分钟的坐标数据，用于热力图。"""
  import time
  end_time = time.time()
  start_time = end_time - (minutes * 60)
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    cur.execute(
      """
      SELECT x, y, roi_id
      FROM raw_events
      WHERE timestamp >= ? AND timestamp <= ?
      ORDER BY timestamp DESC
      LIMIT 10000;
      """,
      (start_time, end_time),
    )
    rows = cur.fetchall()
  return rows


def get_anomalies(
  min_dwell_time: float,
  min_cluster_size: int,
  db_path: str = DB_PATH,
):
  """获取异常点：停留超时或高密度聚集。"""
  with get_conn(db_path) as conn:
    cur = conn.cursor()
    # 停留超时：从 stay_records 获取，并关联 raw_events 获取坐标
    cur.execute(
      """
      SELECT sr.person_id, sr.roi_id, sr.enter_time, sr.leave_time, sr.total_time,
             re.x, re.y
      FROM stay_records sr
      LEFT JOIN raw_events re ON sr.person_id = re.person_id AND sr.roi_id = re.roi_id
      WHERE sr.total_time >= ?
      GROUP BY sr.person_id, sr.roi_id, sr.enter_time, sr.leave_time, sr.total_time
      ORDER BY sr.total_time DESC
      LIMIT 50;
      """,
      (min_dwell_time,),
    )
    long_stays = cur.fetchall()
    
    # 高密度聚集（cluster_id != -1 且 cluster 内人数 >= min_cluster_size）
    cur.execute(
      """
      SELECT cluster_id, COUNT(DISTINCT person_id) as cnt, AVG(x) as avg_x, AVG(y) as avg_y, MAX(ts) as last_ts
      FROM raw_events
      WHERE cluster_id IS NOT NULL AND cluster_id != -1
      GROUP BY cluster_id
      HAVING cnt >= ?
      ORDER BY cnt DESC
      LIMIT 50;
      """,
      (min_cluster_size,),
    )
    clusters = cur.fetchall()
  
  return long_stays, clusters


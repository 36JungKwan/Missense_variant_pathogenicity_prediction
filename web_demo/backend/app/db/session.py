from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Tạo engine kết nối tới PostgreSQL
engine = create_engine(
    settings.DATABASE_URL, 
    pool_pre_ping=True, 
    pool_size=10, 
    max_overflow=20
)

# Tạo SessionFactory để cấp phát session cho mỗi request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Class Base để các Models kế thừa
Base = declarative_base()

# Dependency dùng trong FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
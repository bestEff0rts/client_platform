from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FinTech MVP"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    database_url: str = "sqlite:///./models.db"
    secret_key: str = "change_me"  # для безопасности
    binance_api_key: str = ""
    binance_api_secret: str = ""

    class Config:
        env_file = ".env"  # параметры можно положить сюда

settings = Settings()

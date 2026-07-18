from services.database_service import DATABASE_PATH, initialize_database, migration_status


if __name__ == "__main__":
    initialize_database()
    print(f"Database ready: {DATABASE_PATH}")
    for migration in migration_status():
        print(
            f"Migration {migration['version']}: {migration['name']} ({migration['applied_at']} UTC)"
        )

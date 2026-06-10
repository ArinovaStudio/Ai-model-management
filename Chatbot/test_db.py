import asyncio
from prisma import Prisma

async def test_connection():
    # Initialize the database client
    db = Prisma()
    
    try:
        # Attempt to connect to the database
        await db.connect()
        print("[SUCCESS] Successfully connected to the XYZ Database!")
        
        # Optional: Try to count the employees just to test a query
        # employee_count = await db.employee.count()
        # print(f"Found {employee_count} employees in the database.")
        
    except Exception as e:
        print("[ERROR] Could not connect to the database.")
        print(e)
        
    finally:
        # Always disconnect when finished
        if db.is_connected():
            await db.disconnect()

# Run the async test
if __name__ == '__main__':
    asyncio.run(test_connection())

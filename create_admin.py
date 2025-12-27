"""
Script untuk membuat user admin/test
"""
from app import app, db
from models import User

def create_admin():
    with app.app_context():
        # Cek apakah admin sudah ada
        admin = User.query.filter_by(username='admin').first()
        
        if admin:
            print("✗ User 'admin' sudah ada!")
            print(f"  Email: {admin.email}")
            print(f"  Created: {admin.created_at}")
            return
        
        # Buat user admin
        admin = User(
            username='admin',
            email='admin@floodprediction.com'
        )
        admin.set_password('admin123')  # GANTI DI PRODUCTION!
        
        db.session.add(admin)
        db.session.commit()
        
        print("✓ User admin berhasil dibuat!")
        print(f"  Username: admin")
        print(f"  Password: admin123")
        print(f"  Email: admin@floodprediction.com")
        print("\n⚠️  PENTING: Ganti password setelah login pertama!")

def create_test_user():
    with app.app_context():
        # Cek apakah test user sudah ada
        test_user = User.query.filter_by(username='testuser').first()
        
        if test_user:
            print("✗ User 'testuser' sudah ada!")
            return
        
        # Buat test user
        test_user = User(
            username='testuser',
            email='test@example.com'
        )
        test_user.set_password('test123')
        
        db.session.add(test_user)
        db.session.commit()
        
        print("✓ Test user berhasil dibuat!")
        print(f"  Username: testuser")
        print(f"  Password: test123")
        print(f"  Email: test@example.com")

def list_users():
    with app.app_context():
        users = User.query.all()
        
        if not users:
            print("✗ Tidak ada user dalam database")
            return
        
        print(f"\n{'='*60}")
        print(f"DAFTAR USER ({len(users)} users)")
        print(f"{'='*60}")
        
        for user in users:
            print(f"\nID: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Created: {user.created_at}")
            print(f"Total Predictions: {len(user.predictions)}")

if __name__ == '__main__':
    print("="*60)
    print("SETUP USER DATABASE")
    print("="*60)
    
    # Pilihan
    print("\n1. Buat admin user")
    print("2. Buat test user")
    print("3. List semua user")
    print("4. Buat keduanya (admin + test)")
    
    choice = input("\nPilih (1-4): ").strip()
    
    if choice == '1':
        create_admin()
    elif choice == '2':
        create_test_user()
    elif choice == '3':
        list_users()
    elif choice == '4':
        create_admin()
        print()
        create_test_user()
        list_users()
    else:
        print("Pilihan tidak valid!")
    
    print("\n" + "="*60)
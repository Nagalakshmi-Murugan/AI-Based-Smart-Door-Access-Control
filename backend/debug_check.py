from database import list_users_with_encodings
from face_module import load_known_users, get_known_users_count

users = list_users_with_encodings('smart_door.db')
print('DB users:', len(users))

for u in users:
    enc = u.get('face_encoding', [])
    print(f"  name={u['name']} | student_id={u.get('student_id')} | encoding_length={len(enc)}")

count = load_known_users(users)
print('Loaded into memory:', count)
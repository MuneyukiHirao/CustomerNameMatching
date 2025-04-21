import os
import pytest
from fastapi.testclient import TestClient

# Override DATABASE_URL before importing app and db
os.environ['DATABASE_URL'] = 'sqlite:///./test_db.sqlite'

from backend.main import app
from backend.pipeline import init_db
from backend.db import SessionLocal
from backend.models import MatchCandidate, MatchScore

client = TestClient(app)

@pytest.fixture(scope='function')
def setup_and_seed_db():
    # Initialize SQLite DB and schema
    init_db()
    session = SessionLocal()
    # Create candidate and score
    cand = MatchCandidate(cust_idA='A1', cust_idB='B1', vec_cos=0.9, lang_pen=0.0)
    session.add(cand)
    session.flush()  # assign cand.id
    score = MatchScore(cand_id=cand.id, code_sim=1.0, name_sim=1.0, llm_prob=1.0, final_scr=100.0, status='pending')
    session.add(score)
    session.commit()
    return {'cand_id': cand.id}

def test_review_approve(setup_and_seed_db):
    cand_id = setup_and_seed_db['cand_id']
    response = client.put(f'/review/{cand_id}', json={'match': True, 'user': 'tester'})
    assert response.status_code == 200
    assert response.json() == {'status': 'saved'}
    session = SessionLocal()
    updated = session.get(MatchScore, cand_id)
    assert updated.status == 'approved'
    assert updated.action_user == 'tester'

def test_review_reject(setup_and_seed_db):
    cand_id = setup_and_seed_db['cand_id']
    response = client.put(f'/review/{cand_id}', json={'match': False, 'user': 'tester2'})
    assert response.status_code == 200
    session = SessionLocal()
    updated = session.get(MatchScore, cand_id)
    assert updated.status == 'rejected'
    assert updated.action_user == 'tester2'

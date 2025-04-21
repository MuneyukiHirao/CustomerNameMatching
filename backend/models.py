from sqlalchemy import Column, String, Unicode, Float, ForeignKey, DateTime, LargeBinary
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.orm import relationship
import uuid
from db import Base
import os
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

class GUID(TypeDecorator):
    """Platform-independent GUID type."""
    impl = CHAR
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID())
        else:
            return dialect.type_descriptor(CHAR(36))
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == 'postgresql':
            return str(value)
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(value))
        return str(value)
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return uuid.UUID(value)

class CustomerStdA(Base):
    __tablename__ = 'customers_std_A'
    cust_id = Column(String(50), primary_key=True)
    lang = Column(String(2))
    norm_name = Column(Unicode(200))
    norm_code = Column(String(50))
    vector = Column(LargeBinary)
    raw_name = Column(Unicode(200))
    raw_code = Column(String(50))

class CustomerStdB(Base):
    __tablename__ = 'customers_std_B'
    cust_id = Column(String(50), primary_key=True)
    lang = Column(String(2))
    norm_name = Column(Unicode(200))
    norm_code = Column(String(50))
    vector = Column(LargeBinary)
    raw_name = Column(Unicode(200))
    raw_code = Column(String(50))

class MatchCandidate(Base):
    __tablename__ = 'match_candidates'
    id = Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    cust_idA = Column(String(50))
    cust_idB = Column(String(50))
    vec_cos = Column(Float)
    lang_pen = Column(Float)
    scores = relationship('MatchScore', back_populates='candidate')

class MatchScore(Base):
    __tablename__ = 'match_scores'
    cand_id = Column(GUID(), ForeignKey('match_candidates.id'), primary_key=True)
    code_sim = Column(Float)
    name_sim = Column(Float)
    llm_prob = Column(Float)
    final_scr = Column(Float)
    status = Column(String(10))
    action_user = Column(String(64))
    action_time = Column(DateTime)
    candidate = relationship('MatchCandidate', back_populates='scores')

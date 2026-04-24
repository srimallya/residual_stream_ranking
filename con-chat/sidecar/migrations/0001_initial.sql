create table if not exists conversations (
  id text primary key,
  title text not null,
  current_tokens integer not null,
  max_tokens integer not null,
  created_at text not null,
  updated_at text not null
);

create table if not exists turns (
  id text primary key,
  conversation_id text not null,
  role text not null,
  text text not null,
  token_count integer not null,
  created_at text not null
);

create table if not exists memory_objects (
  id text primary key,
  conversation_id text not null,
  kind text not null,
  tier text not null,
  byte_size integer not null,
  source_turn_start text not null,
  source_turn_end text not null,
  summary text not null,
  last_used_at text not null,
  created_at text not null
);

create table if not exists memory_edges (
  id text primary key,
  conversation_id text not null,
  from_id text not null,
  to_id text not null,
  edge_type text not null,
  created_at text not null
);

create table if not exists ledger_events (
  id text primary key,
  conversation_id text not null,
  object_id text,
  event_type text not null,
  payload_json text not null,
  created_at text not null
);

create table if not exists settings (
  key text primary key,
  value_json text not null,
  updated_at text not null
);

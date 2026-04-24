alter table memory_objects add column replay_layer integer not null default 34;
alter table memory_objects add column payload_blob blob not null default x'';
alter table memory_objects add column fallback_text text not null default '';
alter table memory_objects add column retrieval_key text not null default '';
alter table memory_objects add column graph_metadata_json text not null default '{}';
alter table memory_objects add column pinned integer not null default 0;

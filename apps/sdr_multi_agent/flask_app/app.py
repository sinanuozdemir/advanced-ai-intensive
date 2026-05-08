#!/usr/bin/env python3
"""
Flask app with Generic LangGraph React Agent and MCP integration
Now with Celery async task support
"""

import os
import asyncio
import json
import glob
import threading
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage

from agent_builder import GenericAgent
from memory_agent import make_agent, MemoryAgent, memory_scope_for_config, KNOWN_MEMORY_SCOPES, memory_stores_snapshot  # factory picks GenericAgent or MemoryAgent
from celery_app import process_chat_task, get_available_tools_task, get_conversation_history_task, get_task_result
from subtask_log import SubtaskLog

_key_ok = bool(os.getenv("OPENROUTER_API_KEY"))
print(f"OPENROUTER_API_KEY: {'set' if _key_ok else 'MISSING — add to .env'}")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'

USE_SUPERVISOR = os.getenv('USE_SUPERVISOR', '1').lower() in {'1', 'true', 'yes'}

# Global agent instance for synchronous endpoints. The default chat agent is
# the SDR supervisor; the dropdown still routes to per-config GenericAgents
# for debugging individual sub-agents.
if USE_SUPERVISOR:
    from supervisor import SupervisorAgent
    generic_agent = SupervisorAgent()
    print("🧭 Default agent: SupervisorAgent (set USE_SUPERVISOR=0 to disable)")
else:
    generic_agent = make_agent()

_subtask_log = SubtaskLog()

# Global agent instances cache
agent_cache = {}

# LangGraph AsyncSqliteSaver binds to the asyncio loop it was created on. Sync
# Flask handlers were doing `new_event_loop()` per request, so initialize()
# attached the saver to loop A and chat() ran on loop B → "bound to a different
# event loop". Use one loop per process and serialize access.
_flask_agent_loop: asyncio.AbstractEventLoop | None = None
_flask_agent_async_lock = threading.Lock()


def run_agent_coro(coro):
    """Run *coro* on a shared event loop (thread-safe)."""
    global _flask_agent_loop
    with _flask_agent_async_lock:
        if _flask_agent_loop is None or _flask_agent_loop.is_closed():
            _flask_agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_flask_agent_loop)
        return _flask_agent_loop.run_until_complete(coro)


def get_agent(agent_config: str = None):
    """Get or create an agent instance.

    With no `agent_config`, returns the default chat agent (the supervisor
    when `USE_SUPERVISOR=1`, else `make_agent()`). With an explicit config,
    returns a `GenericAgent` for that config — used by the UI dropdown to
    talk directly to a sub-agent for debugging.
    """
    config_key = agent_config or "default"

    if config_key not in agent_cache:
        if agent_config:
            use_mem = os.environ.get("USE_MEMORY_AGENT", "").lower() in {"1", "true", "yes"}
            if use_mem:
                scope = memory_scope_for_config(agent_config)
                agent_cache[config_key] = MemoryAgent(
                    config_path=agent_config,
                    memory_scope=scope,
                )
            else:
                agent_cache[config_key] = GenericAgent(config_path=agent_config)
        else:
            agent_cache[config_key] = generic_agent

    return agent_cache[config_key]

def discover_agent_configs():
    """Dynamically discover agent configuration files"""
    configs = {}
    
    # Get the directory where the Flask app is running
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all JSON files that could be agent configs
    config_patterns = [
        os.path.join(app_dir, 'agent_config*.json'),  # Original pattern
        os.path.join(app_dir, '*_config.json'),       # Files ending with _config.json
        os.path.join(app_dir, '*_agent.json'),        # Files ending with _agent.json
        os.path.join(app_dir, 'sdr*.json'),           # SDR specific configs
        os.path.join(app_dir, 'qualifying*.json')     # Qualifying specific configs
    ]
    
    config_files = []
    for pattern in config_patterns:
        config_files.extend(glob.glob(pattern))
    
    # Remove duplicates while preserving order
    config_files = list(dict.fromkeys(config_files))
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Extract filename for the key
            filename = os.path.basename(config_file)
            
            # Extract name and description from config
            name = config_data.get('name', filename.replace('.json', '').replace('_', ' ').title())
            description = config_data.get('description', 'No description available')
            
            configs[filename] = {
                'name': name,
                'description': description,
                'filename': filename
            }
            
        except Exception as e:
            print(f"Error loading config file {config_file}: {str(e)}")
            continue
    
    return configs

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/agent-configs')
def get_agent_configs():
    """Get available agent configurations"""
    try:
        configs = discover_agent_configs()
        return jsonify({
            "success": True,
            "configs": configs
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": generic_agent.initialized,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "flask": "running",
            "celery": "available",
            "rabbitmq": "connected"
        }
    })

# =============================================================================
# SYNCHRONOUS ENDPOINTS
# =============================================================================

@app.route('/api/initialize', methods=['POST'])
def initialize_agent():
    """Initialize the generic agent"""
    try:
        data = request.get_json() or {}
        agent_config = data.get('agent_config')
        
        # Initialize the agent with the specified config
        agent = get_agent(agent_config)

        run_agent_coro(agent.initialize())
        tools = run_agent_coro(agent.get_available_tools())
        
        return jsonify({
            "success": True,
            "message": "Generic Agent initialized successfully with persistent memory",
            "tools_count": len(tools),
            "tools": tools
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the generic agent (synchronous)"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data.get('message')
    conversation_id = data.get('conversation_id', data.get('thread_id', 'default'))
    agent_config = data.get('agent_config')
    
    print(f"🔍 Sync Chat - Conversation ID: {conversation_id}, Agent Config: {agent_config}")
    
    # Get the appropriate agent instance
    agent = get_agent(agent_config)

    result = run_agent_coro(agent.chat(message, conversation_id))
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 500

@app.route('/api/tools')
def get_tools():
    """Get available tools (synchronous)"""
    try:
        agent_config = request.args.get('agent_config')
        agent = get_agent(agent_config)

        tools = run_agent_coro(agent.get_available_tools())
        return jsonify({
            "success": True,
            "tools": tools
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>')
def get_conversation(conversation_id):
    """Get conversation history (synchronous)"""
    try:
        history = run_agent_coro(generic_agent.get_conversation_history(conversation_id))
        
        return jsonify({
            "conversation_id": conversation_id,
            "history": history
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """Clear conversation history (synchronous)"""
    try:
        success = run_agent_coro(generic_agent.clear_conversation(conversation_id))
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Conversation {conversation_id} cleared successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to clear conversation"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =============================================================================
# SUPERVISOR SUB-TASK PANEL
# =============================================================================

@app.route('/api/subtasks/<conversation_id>')
def list_subtasks(conversation_id):
    """List sub-tasks dispatched by the supervisor for a conversation.

    On every poll we reconcile any non-terminal rows against the Celery
    result backend and write the new state back into the local SQLite log,
    so the UI sees the latest status without having to query Celery itself.
    """
    try:
        rows = _subtask_log.list_for_conversation(conversation_id)
        for row in rows:
            if row['status'] not in ('PENDING', 'PROCESSING'):
                continue
            r = get_task_result(row['task_id'])
            new_state = r.get('state', row['status'])
            summary = None
            if new_state == 'SUCCESS':
                payload = r.get('result') or {}
                if isinstance(payload, dict):
                    summary = payload.get('response', '')
            elif new_state in ('FAILURE', 'ERROR'):
                summary = r.get('error')
                if not summary:
                    payload = r.get('result') or {}
                    if isinstance(payload, dict):
                        summary = payload.get('error', 'unknown')
            if new_state != row['status'] or summary is not None:
                _subtask_log.update_status(row['task_id'], new_state, result_summary=summary)
                row['status'] = new_state
                if summary is not None:
                    row['result_summary'] = summary
        return jsonify({'success': True, 'conversation_id': conversation_id, 'subtasks': rows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/subtasks/<task_id>/trace')
def subtask_trace(task_id):
    """Return the full message trace (system / human / ai / tool calls /
    tool results) for the sub-agent run that handled `task_id`.

    Reads LangGraph state via the same MemoryAgent (same checkpointer / DB)
    that ran the task — so it works across processes (Celery wrote it,
    Flask reads it) as long as DATABASE_URL points at the shared backend.
    """
    try:
        row = _subtask_log.get(task_id)
        if not row:
            return jsonify({'success': False, 'error': 'unknown task_id'}), 404
        thread_id = row.get('thread_id')
        if not thread_id:
            return jsonify({
                'success': False,
                'error': 'no thread_id recorded for this sub-task (older row)',
                'subtask': row,
            }), 404
        agent = get_agent(row['agent_config'])
        run_agent_coro(agent.initialize())
        messages = run_agent_coro(agent.get_full_trace(thread_id))
        return jsonify({
            'success': True,
            'task_id': task_id,
            'thread_id': thread_id,
            'agent_config': row['agent_config'],
            'agent_slug': row['agent_slug'],
            'message': row['message'],
            'status': row['status'],
            'messages': messages,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/threads/<thread_id>/trace')
def thread_trace(thread_id):
    """Phase A: load a trace by thread_id directly (e.g. for the supervisor's
    own thread). Pass `?config=<agent_config.json>` to pick the agent;
    defaults to the supervisor / default agent."""
    try:
        agent_config = request.args.get('config')
        agent = get_agent(agent_config) if agent_config else generic_agent
        run_agent_coro(agent.initialize())
        messages = run_agent_coro(agent.get_full_trace(thread_id))
        return jsonify({
            'success': True,
            'thread_id': thread_id,
            'agent_config': agent_config or 'default',
            'messages': messages,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# ASYNCHRONOUS ENDPOINTS
# =============================================================================

@app.route('/api/async/chat', methods=['POST'])
def start_chat_task():
    """Start an async chat task and return task ID"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data.get('message')
    thread_id = data.get('thread_id', data.get('conversation_id', 'default'))
    agent_config = data.get('agent_config')  # Optional agent configuration
    
    try:
        # Start the Celery task with optional agent config
        task = process_chat_task.delay(message, thread_id, agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Chat task started",
            "thread_id": thread_id,
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start chat task: {str(e)}"
        }), 500

@app.route('/api/async/conversations/<thread_id>', methods=['POST'])
def start_conversation_history_task(thread_id):
    """Start an async task to get conversation history"""
    data = request.get_json() or {}
    agent_config = data.get('agent_config')  # Optional agent configuration
    
    try:
        # Start the Celery task with optional agent config
        task = get_conversation_history_task.delay(thread_id, agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": f"Conversation history task started for thread {thread_id}",
            "thread_id": thread_id,
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start conversation history task: {str(e)}"
        }), 500

@app.route('/api/async/tools', methods=['GET'])
def start_tools_task():
    """Start an async task to get available tools"""
    agent_config = request.args.get('agent_config')
    
    try:
        # Start the Celery task with optional agent config
        task = get_available_tools_task.delay(agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Tools task started",
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start tools task: {str(e)}"
        }), 500

@app.route('/api/async/tasks/<task_id>')
def get_task_status(task_id):
    """Get the status and result of an async task"""
    try:
        result = get_task_result(task_id)
        
        # Determine HTTP status code based on task state
        if result['state'] in ['PENDING', 'PROCESSING']:
            status_code = 202  # Still processing
        elif result['state'] == 'SUCCESS':
            status_code = 200  # Complete
        else:
            status_code = 500  # Error
            
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            "state": "ERROR",
            "error": f"Failed to get task status: {str(e)}",
            "task_id": task_id
        }), 500

@app.route('/api/memory/stores')
def memory_stores():
    """Inspect semantic / episodic / procedural stores (per scope or all)."""
    try:
        scope = (request.args.get("scope") or "all").strip().lower()
        if scope == "all":
            return jsonify({
                "success": True,
                "scopes": {s: memory_stores_snapshot(s) for s in KNOWN_MEMORY_SCOPES},
            })
        if scope not in KNOWN_MEMORY_SCOPES:
            return jsonify({
                "success": False,
                "error": f"unknown scope; use one of {list(KNOWN_MEMORY_SCOPES)} or all",
            }), 400
        return jsonify({"success": True, "stores": memory_stores_snapshot(scope)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/memory/reflect', methods=['POST'])
def memory_reflect():
    """Run end-of-thread reflection for the MemoryAgent backing this session."""
    try:
        data = request.get_json() or {}
        thread_id = data.get("thread_id") or data.get("conversation_id")
        if not thread_id:
            return jsonify({"success": False, "error": "thread_id is required"}), 400
        agent_config = data.get("agent_config")
        rubric_score = float(data.get("rubric_score", 0.0))
        agent = get_agent(agent_config if agent_config else None)
        if not isinstance(agent, MemoryAgent):
            return jsonify({
                "success": False,
                "error": "Selected agent is not MemoryAgent; set USE_MEMORY_AGENT=1",
            }), 400
        reflect_result = run_agent_coro(agent.end_thread(thread_id, rubric_score))
        return jsonify({"success": True, "reflect": reflect_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/async/tasks')
def list_task_endpoints():
    """List available async task endpoints"""
    return jsonify({
        "async_endpoints": {
            "start_chat": {
                "method": "POST",
                "url": "/api/async/chat",
                "description": "Start async chat with agent",
                "payload": {
                    "message": "Your question here",
                    "thread_id": "optional_thread_id"
                }
            },
            "start_conversation_history": {
                "method": "POST",
                "url": "/api/async/conversations/{thread_id}",
                "description": "Start async conversation history fetch"
            },
            "get_tools": {
                "method": "GET",
                "url": "/api/async/tools",
                "description": "Get available tools for specified agent config"
            },
            "get_task_status": {
                "method": "GET",
                "url": "/api/async/tasks/{task_id}",
                "description": "Get task status and result"
            }
        },
        "task_states": {
            "PENDING": "Task is waiting to be processed",
            "PROCESSING": "Task is currently being processed",
            "SUCCESS": "Task completed successfully",
            "FAILURE": "Task failed with error",
            "ERROR": "System error occurred"
        }
    })

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Agent Flask App with Celery support...")
    print("🔧 Make sure your services are running:")
    print("   docker-compose up -d")
    print("🌐 Synchronous API: http://localhost:5000 (host port 8080 when launched via docker-compose)")
    print("🌐 Asynchronous API: http://localhost:5000/api/async/ (host port 8080 in docker-compose)")
    print("🐰 RabbitMQ Management: http://localhost:15672 (agent/agent123)")
    print("💾 Conversations are persisted using LangGraph's MemorySaver")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    ) 
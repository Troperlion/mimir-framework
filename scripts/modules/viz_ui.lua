-- Callbacks run only when dispatch() is called by the Lua script.
local UI = {}
function UI.new(scene)
  assert(Mimir.Viz.configure(scene))
  local handlers = {}
  local self = {}
  function self:on(id, callback)
    assert(type(id) == "string" and type(callback) == "function")
    handlers[id] = callback
    return self
  end
  function self:configure(next_scene) assert(Mimir.Viz.configure(next_scene)) end
  function self:dispatch(timeout_ms)
    for _, event in ipairs(Mimir.Viz.poll_events(timeout_ms or 0)) do
      local callback = handlers[event.id] or handlers[event.type]
      if callback then callback(event) end
    end
  end
  return self
end
return UI

-- Mach4: publish active work coordinates (current G54/G55... DRO) to UDP listeners
-- and accept localhost load/run commands from layout_design `python -m app cnc`.
-- Used by: Orbbec CNC stream (62100), layout_design `record-pm` (62101),
--          layout_design `cnc` (127.0.0.1:62110).
-- Install: copy into Mach4Profiles/<profile>/Macros or paste into PLC script.
--
-- Requires LuaSocket in Mach4 (socket.dll + socket/core.dll in Mach4 api/lua folder).
-- Tracking PC must run:
--   orbbec-head-stream-cnc --work-pose-udp-port 62100 ...
--   layout_design: python -m app record-pm --port 62101 ...
-- Same PC as Mach4:
--   layout_design: python -m app cnc status|load|start|hold|stop
--
-- Edit TARGETS to match each consumer on the LAN (one UDP socket per target).
-- Keep pose TARGETS in sync with Orbbec CV `scripts/mach4_work_pose_publisher.lua`.
-- Command socket binds 127.0.0.1 only (do not use 0.0.0.0).

local TARGETS = {
  { ip = "192.168.208.10", port = 62100 }, -- Orbbec head tracking / orbbec-head-stream-cnc
  { ip = "192.168.208.10", port = 62101 }, -- layout_design record-pm
}
local PUBLISH_PERIOD_SEC = 0.05
local CMD_BIND_IP = "127.0.0.1"
local CMD_BIND_PORT = 62110

local inst = mc.mcGetInstance()
local udp_sockets = {}
local lastPublish = 0.0
local cmd_bind_error = nil

-- Global so a PLC/script reload can close the previous OS bind (locals reset,
-- but the port stays held until that userdata is closed).
local function close_cmd_sock()
  local sock = LayoutDesignCncCmdSock
  LayoutDesignCncCmdSock = nil
  if sock ~= nil then
    pcall(function()
      sock:close()
    end)
  end
end

close_cmd_sock()

local function axis_pos(axisConst)
  -- mcAxisGetPos returns the active work coordinate (not machine coords).
  return mc.mcAxisGetPos(inst, axisConst)
end

local function ensure_udp()
  if next(udp_sockets) ~= nil then
    return true
  end
  local ok, socket = pcall(require, "socket")
  if not ok then
    mc.mcCntlSetLastError(inst, "work pose UDP: LuaSocket not available")
    return false
  end
  for i, target in ipairs(TARGETS) do
    local udp = socket.udp()
    udp:setpeername(target.ip, target.port)
    udp_sockets[i] = udp
  end
  return true
end

function PublishWorkPoseUdp()
  local now = os.clock()
  if (now - lastPublish) < PUBLISH_PERIOD_SEC then
    return
  end
  lastPublish = now
  if not ensure_udp() then
    return
  end

  local x = axis_pos(mc.X_AXIS)
  local y = axis_pos(mc.Y_AXIS)
  local z = axis_pos(mc.Z_AXIS)
  local b = axis_pos(mc.B_AXIS)
  local c = axis_pos(mc.C_AXIS)

  local payload = string.format(
    '{"coord":"work","units":"mm","x":%.4f,"y":%.4f,"z":%.4f,"b":%.4f,"c":%.4f}',
    x, y, z, b, c
  )
  for _, udp in ipairs(udp_sockets) do
    udp:send(payload)
  end
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
  s = s:gsub('"', '\\"')
  s = s:gsub("\r", "\\r")
  s = s:gsub("\n", "\\n")
  return s
end

local function json_get_string(payload, key)
  local pat = '"' .. key .. '"%s*:%s*"'
  local _, start_at = payload:find(pat)
  if not start_at then
    return nil
  end
  local i = start_at + 1
  local out = {}
  while i <= #payload do
    local ch = payload:sub(i, i)
    if ch == "\\" then
      local nxt = payload:sub(i + 1, i + 1)
      if nxt == "n" then
        out[#out + 1] = "\n"
      elseif nxt == "r" then
        out[#out + 1] = "\r"
      elseif nxt == "t" then
        out[#out + 1] = "\t"
      else
        out[#out + 1] = nxt
      end
      i = i + 2
    elseif ch == '"' then
      return table.concat(out)
    else
      out[#out + 1] = ch
      i = i + 1
    end
  end
  return nil
end

local function state_label()
  local st = nil
  pcall(function()
    st = mc.mcCntlGetState(inst)
  end)
  if st == nil then
    return "unknown"
  end
  local names = {
    { "MC_STATE_IDLE", "idle" },
    { "MC_STATE_HOLD", "hold" },
    { "MC_STATE_JOG", "jog" },
    { "MC_STATE_RUNNING", "running" },
    { "MC_STATE_RUN", "running" },
    { "MC_STATE_HOMING", "homing" },
  }
  for _, pair in ipairs(names) do
    local const_name, label = pair[1], pair[2]
    if mc[const_name] ~= nil and st == mc[const_name] then
      return label
    end
  end
  return tostring(st)
end

local function is_idle()
  local st = nil
  pcall(function()
    st = mc.mcCntlGetState(inst)
  end)
  if st == nil then
    return false
  end
  if mc.MC_STATE_IDLE ~= nil then
    return st == mc.MC_STATE_IDLE
  end
  return st == 0
end

local function machine_enabled()
  local ok, hsig = pcall(mc.mcSignalGetHandle, inst, mc.OSIG_MACHINE_ENABLED)
  if not ok or hsig == nil then
    return false
  end
  local ok2, val = pcall(mc.mcSignalGetState, hsig)
  if not ok2 then
    return false
  end
  return val == 1
end

local function gcode_filename()
  local ok, name = pcall(mc.mcCntlGetGcodeFileName, inst)
  if ok and type(name) == "string" then
    return name
  end
  return ""
end

local function api_ok(rc)
  if rc == nil then
    return true
  end
  if mc.MERROR_NOERROR ~= nil then
    return rc == mc.MERROR_NOERROR
  end
  return rc == 0
end

local function work_pose()
  local x, y, z, b, c = 0.0, 0.0, 0.0, 0.0, 0.0
  pcall(function()
    x = axis_pos(mc.X_AXIS)
    y = axis_pos(mc.Y_AXIS)
    z = axis_pos(mc.Z_AXIS)
    b = axis_pos(mc.B_AXIS)
    c = axis_pos(mc.C_AXIS)
  end)
  return x, y, z, b, c
end

local function make_ack(ok, id, cmd, err)
  local x, y, z, b, c = work_pose()
  return table.concat({
    '{"ok":', ok and "true" or "false",
    ',"id":"', json_escape(id),
    ',"cmd":"', json_escape(cmd),
    ',"error":"', json_escape(err or ""),
    ',"state":"', json_escape(state_label()),
    ',"enabled":', machine_enabled() and "true" or "false",
    ',"file":"', json_escape(gcode_filename()),
    ',"x":', string.format("%.4f", x),
    ',"y":', string.format("%.4f", y),
    ',"z":', string.format("%.4f", z),
    ',"b":', string.format("%.4f", b),
    ',"c":', string.format("%.4f", c),
    "}",
  })
end

local function file_readable(path)
  local f = io.open(path, "r")
  if not f then
    return false
  end
  f:close()
  return true
end

local function handle_cmd(payload)
  local cmd = json_get_string(payload, "cmd") or ""
  local id = json_get_string(payload, "id") or ""
  if cmd == "" then
    return make_ack(false, id, cmd, "missing cmd")
  end

  if cmd == "status" then
    return make_ack(true, id, cmd, "")
  end

  if cmd == "hold" then
    local ok, rc = pcall(mc.mcCntlFeedHold, inst)
    if not ok then
      return make_ack(false, id, cmd, tostring(rc))
    end
    if not api_ok(rc) then
      return make_ack(false, id, cmd, "feed hold rc=" .. tostring(rc))
    end
    return make_ack(true, id, cmd, "")
  end

  if cmd == "stop" then
    local fn = mc.mcCntlCycleStop or mc.mcCntlStop
    local ok, rc = pcall(fn, inst)
    if not ok then
      return make_ack(false, id, cmd, tostring(rc))
    end
    if not api_ok(rc) then
      return make_ack(false, id, cmd, "cycle stop rc=" .. tostring(rc))
    end
    return make_ack(true, id, cmd, "")
  end

  if cmd == "load" or cmd == "start" then
    if not machine_enabled() then
      return make_ack(false, id, cmd, "machine not enabled")
    end
    if not is_idle() then
      return make_ack(false, id, cmd, "not idle (state=" .. state_label() .. ")")
    end
  end

  if cmd == "load" then
    local path = json_get_string(payload, "path") or ""
    if path == "" then
      return make_ack(false, id, cmd, "load requires path")
    end
    if not file_readable(path) then
      return make_ack(false, id, cmd, "file not found")
    end
    local ok, rc = pcall(mc.mcCntlLoadGcodeFile, inst, path)
    if not ok then
      return make_ack(false, id, cmd, tostring(rc))
    end
    if not api_ok(rc) then
      return make_ack(false, id, cmd, "load rc=" .. tostring(rc))
    end
    return make_ack(true, id, cmd, "")
  end

  if cmd == "start" then
    local ok, rc = pcall(mc.mcCntlCycleStart, inst)
    if not ok then
      return make_ack(false, id, cmd, tostring(rc))
    end
    if not api_ok(rc) then
      return make_ack(false, id, cmd, "cycle start rc=" .. tostring(rc))
    end
    return make_ack(true, id, cmd, "")
  end

  return make_ack(false, id, cmd, "unknown cmd")
end

local function ensure_cmd_udp()
  if LayoutDesignCncCmdSock ~= nil then
    return true
  end
  local ok, socket = pcall(require, "socket")
  if not ok then
    mc.mcCntlSetLastError(inst, "cnc command UDP: LuaSocket not available")
    return false
  end
  local udp = socket.udp()
  udp:settimeout(0)
  pcall(function()
    udp:setoption("reuseaddr", true)
  end)
  local bind_ok, err = udp:setsockname(CMD_BIND_IP, CMD_BIND_PORT)
  if bind_ok == nil then
    pcall(function()
      udp:close()
    end)
    local msg = "cnc command UDP: bind failed " .. tostring(err)
    if cmd_bind_error ~= msg then
      cmd_bind_error = msg
      mc.mcCntlSetLastError(inst, msg)
    end
    return false
  end
  cmd_bind_error = nil
  LayoutDesignCncCmdSock = udp
  return true
end

function PollCncCommandUdp()
  if not ensure_cmd_udp() then
    return
  end
  local sock = LayoutDesignCncCmdSock
  local data, ip, port = sock:receivefrom()
  if not data then
    return
  end
  local ok, ack = pcall(handle_cmd, data)
  if not ok then
    ack = make_ack(false, "", "", tostring(ack))
  end
  sock:sendto(ack, ip, port)
end

-- Call PublishWorkPoseUdp() and PollCncCommandUdp() from the profile PLC
-- script each cycle. PollCncCommandUdp processes at most one datagram per call.
-- Paste this file once (not Macros plus a second full copy in PLC). If bind
-- still fails after a reload, Disable then Enable Mach4 to drop the old socket.

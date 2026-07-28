-- Mach4: publish active work coordinates (current G54/G55... DRO) to UDP listeners.
-- Used by: Orbbec CNC stream (62100) and layout_design `record-pm` (62101).
-- Install: copy into Mach4Profiles/<profile>/Macros or paste into PLC script.
--
-- Requires LuaSocket in Mach4 (socket.dll + socket/core.dll in Mach4 api/lua folder).
-- Tracking PC must run:
--   orbbec-head-stream-cnc --work-pose-udp-port 62100 ...
--   layout_design: python -m app record-pm --port 62101 ...
--
-- Edit TARGETS to match each consumer on the LAN (one UDP socket per target).
-- Keep this file in sync with Orbbec CV `scripts/mach4_work_pose_publisher.lua`.

local TARGETS = {
  { ip = "192.168.208.10", port = 62100 }, -- Orbbec head tracking / orbbec-head-stream-cnc
  { ip = "192.168.208.10", port = 62101 }, -- layout_design record-pm
}
local PUBLISH_PERIOD_SEC = 0.05

local inst = mc.mcGetInstance()
local udp_sockets = {}
local lastPublish = 0.0

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

-- Call PublishWorkPoseUdp() from the profile PLC script each cycle.

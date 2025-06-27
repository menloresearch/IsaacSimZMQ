// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#define CARB_EXPORTS

#include <carb/PluginUtils.h>

#include <omni/ext/IExt.h>
#include <omni/graph/core/IGraphRegistry.h>
#include <omni/graph/core/ogn/Database.h>
#include <omni/graph/core/ogn/Registration.h>

// Standard plugin definitions required by Carbonite.
const struct carb::PluginImplDesc pluginImplDesc = { "isaacsim.zmq.bridge.plugin",
                                                     "IsaacSim ZMQ Bridge C++ Ogn Nodes.", "NVIDIA",
                                                     carb::PluginHotReload::eEnabled, "dev" };

// These interface dependencies are required by all OmniGraph node types
CARB_PLUGIN_IMPL_DEPS(omni::graph::core::IGraphRegistry,
                      omni::fabric::IPath,
                      omni::fabric::IToken)

// This macro sets up the information required to register your node type definitions with OmniGraph
DECLARE_OGN_NODES()

namespace isaacsim
{
namespace zmq
{
namespace bridge
{

class OmniGraphIsaacZMQNodeExtension : public omni::ext::IExt
{
public:
    void onStartup(const char* extId) override
    {
        // This macro walks the list of pending node type definitions and registers them with OmniGraph
        INITIALIZE_OGN_NODES()
    }

    void onShutdown() override
    {
        // This macro walks the list of registered node type definitions and deregisters all of them. This is required
        // for hot reload to work.
        RELEASE_OGN_NODES()
    }

private:
};

}
}
}


CARB_PLUGIN_IMPL(pluginImplDesc, isaacsim::zmq::bridge::OmniGraphIsaacZMQNodeExtension)

void fillInterface(isaacsim::zmq::bridge::OmniGraphIsaacZMQNodeExtension& iface)
{
}

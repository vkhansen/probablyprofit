"""
Static Dashboard HTML

A beautiful single-page dashboard that connects to the FastAPI backend.
This is served directly by the FastAPI app, no build step needed.
"""

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>probablyprofit Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .glass { 
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        .gradient-bg {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .card-gradient {
            background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <!-- Header -->
    <header class="border-b border-white/10 px-6 py-4">
        <div class="max-w-7xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🤖</span>
                <h1 class="text-xl font-bold">probablyprofit</h1>
            </div>
            <div class="flex items-center space-x-4">
                <div id="status" class="flex items-center space-x-2 text-sm">
                    <span class="w-2 h-2 bg-green-400 rounded-full pulse"></span>
                    <span>Connected</span>
                </div>
                <a href="/docs" target="_blank" class="text-sm text-gray-400 hover:text-white">API Docs →</a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-6 py-8">
        <!-- Stats Row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="glass rounded-xl p-6 card-gradient border border-white/10">
                <div class="text-gray-400 text-sm mb-1">Balance</div>
                <div id="balance" class="text-3xl font-bold text-green-400">$—</div>
            </div>
            <div class="glass rounded-xl p-6 card-gradient border border-white/10">
                <div class="text-gray-400 text-sm mb-1">Today's P&L</div>
                <div id="pnl" class="text-3xl font-bold">$—</div>
            </div>
            <div class="glass rounded-xl p-6 card-gradient border border-white/10">
                <div class="text-gray-400 text-sm mb-1">Open Positions</div>
                <div id="positions-count" class="text-3xl font-bold text-blue-400">—</div>
            </div>
            <div class="glass rounded-xl p-6 card-gradient border border-white/10">
                <div class="text-gray-400 text-sm mb-1">Total Trades</div>
                <div id="trades-count" class="text-3xl font-bold text-purple-400">—</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Chart -->
            <div class="lg:col-span-2 glass rounded-xl p-6 card-gradient border border-white/10">
                <h2 class="text-lg font-semibold mb-4">Portfolio Value</h2>
                <div class="h-64">
                    <canvas id="portfolioChart"></canvas>
                </div>
            </div>

            <!-- Bot Status -->
            <div class="glass rounded-xl p-6 card-gradient border border-white/10">
                <h2 class="text-lg font-semibold mb-4">Bot Status</h2>
                <div class="space-y-4">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Agent</span>
                        <span id="agent-type" class="font-mono text-sm bg-blue-500/20 px-2 py-1 rounded">—</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Strategy</span>
                        <span id="strategy" class="font-mono text-sm bg-purple-500/20 px-2 py-1 rounded">—</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Uptime</span>
                        <span id="uptime" class="font-mono text-sm">—</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400">Mode</span>
                        <span id="mode" class="font-mono text-sm bg-green-500/20 px-2 py-1 rounded">—</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Positions Table -->
        <div class="mt-6 glass rounded-xl p-6 card-gradient border border-white/10">
            <h2 class="text-lg font-semibold mb-4">Open Positions</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead>
                        <tr class="text-left text-gray-400 text-sm border-b border-white/10">
                            <th class="pb-3">Market</th>
                            <th class="pb-3">Side</th>
                            <th class="pb-3">Size</th>
                            <th class="pb-3">Entry</th>
                            <th class="pb-3">Current</th>
                            <th class="pb-3">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="positions-table" class="text-sm">
                        <tr>
                            <td colspan="6" class="py-8 text-center text-gray-400">No open positions</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="mt-6 glass rounded-xl p-6 card-gradient border border-white/10">
            <h2 class="text-lg font-semibold mb-4">Recent Trades</h2>
            <div id="trades-list" class="space-y-2 text-sm">
                <div class="text-gray-400 text-center py-4">No trades yet</div>
            </div>
        </div>
    </main>

    <script>
        // Chart setup
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { 
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#9ca3af' }
                    },
                    y: { 
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#9ca3af', callback: v => '$' + v }
                    }
                }
            }
        });

        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            document.getElementById('status').querySelector('span:last-child').textContent = 'Connected';
        };
        
        ws.onclose = () => {
            document.getElementById('status').innerHTML = `
                <span class="w-2 h-2 bg-red-400 rounded-full"></span>
                <span>Disconnected</span>
            `;
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        // Fetch initial data
        async function fetchData() {
            try {
                const [statusRes, positionsRes] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/positions')
                ]);
                
                const status = await statusRes.json();
                const positions = await positionsRes.json();
                
                updateStatus(status);
                updatePositions(positions);
            } catch (e) {
                console.error('Failed to fetch data:', e);
            }
        }

        function updateStatus(data) {
            document.getElementById('agent-type').textContent = data.agent_type || 'Unknown';
            document.getElementById('strategy').textContent = data.strategy || 'Unknown';
            document.getElementById('mode').textContent = data.dry_run ? 'Dry Run' : 'Live';
            document.getElementById('uptime').textContent = formatUptime(data.uptime_seconds || 0);
        }

        function updatePositions(positions) {
            const tbody = document.getElementById('positions-table');
            document.getElementById('positions-count').textContent = positions.length;
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="py-8 text-center text-gray-400">No open positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(p => `
                <tr class="border-b border-white/5 hover:bg-white/5">
                    <td class="py-3">${p.market_id?.slice(0, 20)}...</td>
                    <td class="py-3">${p.outcome}</td>
                    <td class="py-3">${p.size}</td>
                    <td class="py-3">$${p.avg_price?.toFixed(2)}</td>
                    <td class="py-3">$${p.current_price?.toFixed(2)}</td>
                    <td class="py-3 ${p.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        ${p.pnl >= 0 ? '+' : ''}$${p.pnl?.toFixed(2)}
                    </td>
                </tr>
            `).join('');
        }

        function updateDashboard(data) {
            if (data.balance !== undefined) {
                document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
            }
            if (data.pnl !== undefined) {
                const pnlEl = document.getElementById('pnl');
                pnlEl.textContent = `${data.pnl >= 0 ? '+' : ''}$${data.pnl.toFixed(2)}`;
                pnlEl.className = `text-3xl font-bold ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
            }
            if (data.trades_count !== undefined) {
                document.getElementById('trades-count').textContent = data.trades_count;
            }
            if (data.portfolio_value !== undefined) {
                const now = new Date().toLocaleTimeString();
                chart.data.labels.push(now);
                chart.data.datasets[0].data.push(data.portfolio_value);
                
                // Keep last 20 points
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
                chart.update('none');
            }
        }

        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h}h ${m}m ${s}s`;
        }

        // Initial fetch
        fetchData();
        
        // Refresh every 30 seconds
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
'''

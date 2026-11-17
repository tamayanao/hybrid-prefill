import AppKit
import SwiftUI

@main
struct HybridPrefill: App {
    @StateObject private var workflow = Workflow()

    var body: some Scene {
        WindowGroup("Hybrid Prefill") {
            ContentView(workflow: workflow)
                .frame(minWidth: 570, minHeight: 380)
        }
        .defaultSize(width: 660, height: 460)
    }
}

struct ContentView: View {
    @ObservedObject var workflow: Workflow

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Hybrid Prefill").font(.largeTitle.bold())
                Text("v.0.0.1 検証テスト版").font(.callout).foregroundStyle(.secondary)
            }
            Text("Mac StudioとRTX搭載PCでDeepSeekの入力処理を分担します。")
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button(workflow.running ? "停止" : "起動") {
                    workflow.running ? workflow.stop() : workflow.start()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                Button("設定ファイルを開く") { workflow.openSettings() }
                    .disabled(workflow.running)
                Spacer()
                if workflow.running && !workflow.ready { ProgressView().controlSize(.small) }
                Text(workflow.status).font(.callout)
            }
            Divider()
            ScrollView {
                Text(workflow.messages.isEmpty ? "起動状況はここに表示されます。" : workflow.messages)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(workflow.messages.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }.padding(24)
    }
}

@MainActor
final class Workflow: ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var ready = false
    @Published private(set) var status = "停止中"
    @Published private(set) var messages = ""

    private var active: Process?
    private var pending = Data()
    private var runID = UUID()
    private var observer: NSObjectProtocol?
    private var config: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("HybridPrefill", isDirectory: true)
            .appendingPathComponent("launcher.json")
    }

    init() {
        observer = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                if let child = self?.active, child.isRunning { child.terminate() }
            }
        }
    }

    func openSettings() {
        do {
            try FileManager.default.createDirectory(at: config.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            if !FileManager.default.fileExists(atPath: config.path) {
                let sample = Bundle.main.url(forResource: "launcher.example", withExtension: "json")!
                try Data(contentsOf: sample).write(to: config, options: [.atomic])
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: config.path)
            }
            NSWorkspace.shared.open(config)
        } catch {
            append("設定を開けません: \(error.localizedDescription)")
        }
    }

    func start() {
        guard active == nil else { return }
        guard let script = Bundle.main.url(forResource: "launch", withExtension: "py") else {
            append("アプリの実行ファイルが見つかりません。")
            return
        }
        guard FileManager.default.fileExists(atPath: config.path) else {
            openSettings()
            append("初回設定を保存してから起動してください。")
            return
        }
        let child = Process()
        let pipe = Pipe()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        child.arguments = ["-B", "-u", script.path, config.path]
        child.standardOutput = pipe
        child.standardError = pipe
        let id = UUID()
        runID = id
        pending = Data()
        messages = ""
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil }
            Task { @MainActor [weak self] in
                guard let self, self.runID == id else { return }
                self.receive(data)
            }
        }
        child.terminationHandler = { [weak self] ended in
            let code = ended.terminationStatus
            Task { @MainActor [weak self] in
                guard let self, self.runID == id else { return }
                self.active = nil
                self.running = false
                self.ready = false
                self.status = code == 0 ? "停止中" : "起動エラー"
            }
        }
        do {
            try child.run()
            active = child
            running = true
            status = "接続中"
        } catch {
            append(error.localizedDescription)
            status = "起動エラー"
        }
    }

    func stop() {
        guard let child = active, child.isRunning else { return }
        status = "停止中…"
        child.terminate()
    }

    private func receive(_ data: Data) {
        pending.append(data)
        while let end = pending.firstIndex(of: 10) {
            let line = String(decoding: pending[..<end], as: UTF8.self)
            pending.removeSubrange(...end)
            if line == "READY" {
                ready = true
                status = "稼働中"
            } else {
                append(line)
            }
        }
    }

    private func append(_ line: String) {
        messages = String((messages + line + "\n").suffix(16000))
    }
}

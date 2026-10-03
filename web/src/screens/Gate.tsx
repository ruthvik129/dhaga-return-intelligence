import { Icon, PageHead } from '../ui'

/** Shown on every dashboard screen until this session has classified a file. */
export default function Gate({ screen }: { screen: string }) {
  return (
    <>
      <PageHead title={screen} />
      <section className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
        <div className="stack-sm" style={{ alignItems: 'center', maxWidth: 520, margin: '0 auto' }}>
          <span className="brand-mark" style={{ width: 48, height: 48, borderRadius: 12 }}>
            <Icon name="upload" size={24} />
          </span>
          <h2>Upload a file to build this dashboard</h2>
          <p className="caption">
            Every screen is built only from the returns file you upload and the rows you choose to classify.
            Nothing is shown until then.
          </p>
          <a className="btn primary" href="#/process">
            <Icon name="upload" />
            Go to Upload &amp; Process
          </a>
        </div>
      </section>
    </>
  )
}
